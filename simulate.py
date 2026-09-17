#!/usr/bin/env python
"""
simulate.py
===========
CALVADOS 2 + リン酸化残基モデル (Rauh et al. 2026) による単鎖 IDP の
粗視化 MD シミュレーション。1 残基 = 1 bead。GPU (CUDA) で実行する。

使い方
------
    python simulate.py --config config.yaml --system phos --replica 0
    python simulate.py --config config.yaml --system unphos --replica 2 --gpu 1

特徴
----
* CUDA / OpenCL / CPU を自動で選択 (--platform で強制も可)
* チェックポイントから自動再開 (HTCondor で evict されても続きから走る)
* 進捗を stdout と JSON ステータスファイルの両方に出す
  -> monitor.py で全ジョブの進捗を一覧できる
* レプリカごとに乱数種と初期構造を変える
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import openmm
from openmm import app, unit

import calvados_ff as ff

# 壁時計時間の上限に達して中断したときの終了コード。
# HTCondor 側で on_exit_remove = (ExitCode == 0) にしておくと、
# このコードで抜けたジョブは自動で再キューされ、チェックポイントから続行する。
EXIT_REQUEUE = 85

# ==========================================================================
# 力場パラメータの生成
# ==========================================================================
KB_KJ = 8.3145e-3          # kJ/mol/K


def dielectric_water(T: float) -> float:
    """水の誘電率の経験式 (Akerlof & Oshry 1950; CALVADOS の実装と同一)。"""
    return (5321.0 / T + 233.76 - 0.9297 * T
            + 0.1417e-2 * T * T - 0.8292e-6 * T ** 3)


def yukawa_params(tokens, residues, temperature, ionic_strength, pH):
    """Debye-Hueckel 項のプレファクタと逆デバイ長を返す。

    OpenMM 側のエネルギー式  q1*q2*(exp(-kappa*r)/r - shift)  に合わせて
    q_i = charge_i * sqrt(lB * kT) を各粒子に持たせる。
    """
    r = ff.apply_ph(residues, pH)
    charges = np.array([float(r.loc[t, "q"]) for t in tokens])
    charges[0] += 1.0       # N 末端 (NH3+)
    charges[-1] -= 1.0      # C 末端 (COO-)

    kT = KB_KJ * temperature
    eps_w = dielectric_water(temperature)
    # Bjerrum 長 [nm]
    lB = 1.6021766 ** 2 / (4 * np.pi * 8.854188 * eps_w) * 6.022 * 1000 / kT
    yu_eps = charges * np.sqrt(lB * kT)
    kappa = np.sqrt(8 * np.pi * lB * ionic_strength * 6.022 / 10)
    return yu_eps, kappa, charges, lB


def masses_of(tokens, residues) -> np.ndarray:
    """CALVADOS 流の bead 質量 [Da] (N 末端 +2 = H2, C 末端 +16 = O)。"""
    mw = np.array([float(residues.loc[t, "MW"]) for t in tokens])
    mw[0] += 2.0
    mw[-1] += 16.0
    return mw


# ==========================================================================
# 初期構造
# ==========================================================================
def spiral_positions(n: int, arc: float = 0.38, separation: float = 0.7) -> np.ndarray:
    """アルキメデス螺旋上に n 個の bead を置く (CALVADOS のデフォルト初期配置)。"""
    r, b = arc, separation / (2 * np.pi)
    phi = r / b
    coords = []
    for _ in range(n):
        coords.append([r * np.cos(phi), r * np.sin(phi), 0.0])
        phi += arc / r
        r = b * phi
    return np.array(coords)


def random_walk_positions(n: int, seed: int, bond: float = 0.38,
                          stiffness: float = 0.0, min_dist: float = 0.45,
                          max_try: int = 300) -> np.ndarray:
    """自己回避的なワーム状ランダムウォークで初期構造を作る。

    stiffness (0-1) を上げるほど直線に近い = 伸びた初期構造になる。
    レプリカごとに値を変えることで、コンパクト側と伸長側の両方から
    サンプリングを始められる (収束チェックになる)。
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3))
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    for i in range(1, n):
        for _ in range(max_try):
            w = stiffness * v + (1.0 - stiffness) * rng.normal(size=3)
            nw = np.linalg.norm(w)
            if nw < 1e-8:
                continue
            w /= nw
            cand = pos[i - 1] + bond * w
            lo = max(0, i - 80)          # 直近 80 残基だけ衝突判定 (十分かつ高速)
            if i == 1 or np.min(np.linalg.norm(pos[lo:i - 1] - cand, axis=1),
                                initial=np.inf) > min_dist:
                pos[i] = cand
                v = w
                break
        else:
            pos[i] = pos[i - 1] + bond * w   # 諦めて配置 (最小化で解消される)
            v = w
    return pos - pos.mean(axis=0)


# ==========================================================================
# 進捗レポータ
# ==========================================================================
class ProgressReporter:
    """進捗を stdout と JSON ステータスファイルに書き出す OpenMM レポータ。"""

    def __init__(self, status_path, total_steps, interval, dt_ps,
                 masses, label="", start_step=0, quiet=False):
        self.status_path = Path(status_path)
        self.total_steps = int(total_steps)
        self.interval = int(interval)
        self.dt_ps = float(dt_ps)
        self.masses = np.asarray(masses, dtype=float)
        self.mtot = self.masses.sum()
        self.label = label
        self.quiet = quiet
        self._t0 = time.time()
        self._step0 = int(start_step)
        self._rg_sum = 0.0
        self._rg_n = 0

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        # (steps, positions, velocities, forces, energies, enforcePeriodicBox)
        return (steps, True, False, False, True, False)

    def _rg(self, xyz: np.ndarray) -> float:
        com = (xyz * self.masses[:, None]).sum(axis=0) / self.mtot
        d2 = ((xyz - com) ** 2).sum(axis=1)
        return float(np.sqrt((d2 * self.masses).sum() / self.mtot))

    def report(self, simulation, state):
        step = simulation.currentStep
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        rg = self._rg(xyz)
        self._rg_sum += rg
        self._rg_n += 1

        elapsed = max(time.time() - self._t0, 1e-9)
        done = step - self._step0
        sps = done / elapsed                                  # steps / s
        ns_per_day = sps * self.dt_ps * 1e-3 * 86400
        remaining = self.total_steps - step
        eta_s = remaining / sps if sps > 0 else float("nan")
        pct = 100.0 * step / self.total_steps

        status = dict(
            label=self.label, step=int(step), total_steps=self.total_steps,
            percent=round(pct, 3),
            time_ns=round(step * self.dt_ps * 1e-3, 4),
            total_ns=round(self.total_steps * self.dt_ps * 1e-3, 2),
            ns_per_day=round(ns_per_day, 1),
            eta_hours=round(eta_s / 3600, 2),
            elapsed_hours=round(elapsed / 3600, 3),
            rg_now_nm=round(rg, 4),
            rg_running_mean_nm=round(self._rg_sum / self._rg_n, 4),
            potential_kJ_mol=round(
                state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole), 2),
            pid=os.getpid(), host=os.uname().nodename,
            updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            finished=bool(step >= self.total_steps),
        )
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=1))
        tmp.replace(self.status_path)

        if not self.quiet:
            bar_n = 24
            filled = int(bar_n * pct / 100)
            bar = "#" * filled + "." * (bar_n - filled)
            print(f"[{self.label}] |{bar}| {pct:6.2f}%  "
                  f"{status['time_ns']:9.2f}/{status['total_ns']:g} ns  "
                  f"{ns_per_day:8.1f} ns/day  ETA {eta_s / 3600:6.2f} h  "
                  f"Rg={rg:5.2f} nm (<Rg>={status['rg_running_mean_nm']:5.2f})",
                  flush=True)


# ==========================================================================
# システム構築
# ==========================================================================
def build_topology(tokens, residues):
    import mdtraj as md
    top = md.Topology()
    chain = top.add_chain()
    for t in tokens:
        three = str(residues.loc[t, "three"])
        res = top.add_residue(three, chain)
        top.add_atom("CA", element=md.element.carbon, residue=res)
    for i in range(top.n_atoms - 1):
        top.add_bond(top.atom(i), top.atom(i + 1))
    return top


def build_system(tokens, residues, cfg, box_nm):
    """OpenMM System と力を組み立てる。"""
    temperature = cfg["temperature"]
    ionic = cfg["ionic_strength"]
    pH = cfg["pH"]
    cutoff_lj = cfg["cutoff_lj_nm"]
    cutoff_dh = cfg["cutoff_dh_nm"]
    lj_eps = cfg["eps_factor"] * 4.184        # 0.2 * 4.184 = 0.8368 kJ/mol

    n = len(tokens)
    mw = masses_of(tokens, residues)
    yu_eps, kappa, charges, lB = yukawa_params(
        tokens, residues, temperature, ionic, pH)

    system = openmm.System()
    a = openmm.Vec3(box_nm, 0, 0)
    b = openmm.Vec3(0, box_nm, 0)
    c = openmm.Vec3(0, 0, box_nm)
    system.setDefaultPeriodicBoxVectors(a, b, c)
    for m in mw:
        system.addParticle(m * unit.amu)

    # --- 結合 (harmonic) ---
    hb = openmm.HarmonicBondForce()
    # --- 非イオン性相互作用: Ashbaugh-Hatch (truncated & shifted) ---
    ah_expr = ("select(step(r-2^(1/6)*s),"
               "4*eps*l*((s/r)^12-(s/r)^6-shift),"
               "4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l));"
               "s=0.5*(s1+s2); l=0.5*(l1+l2); "
               "shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6")
    ah = openmm.CustomNonbondedForce(ah_expr)
    ah.addGlobalParameter("eps", lj_eps * unit.kilojoule_per_mole)
    ah.addGlobalParameter("rc", cutoff_lj * unit.nanometer)
    ah.addPerParticleParameter("s")
    ah.addPerParticleParameter("l")

    # --- 静電: 塩遮蔽 Debye-Hueckel (Yukawa) ---
    yu = openmm.CustomNonbondedForce("q*(exp(-kappa*r)/r-shift); q=q1*q2")
    yu.addGlobalParameter("kappa", kappa / unit.nanometer)
    yu.addGlobalParameter("shift", np.exp(-kappa * cutoff_dh) / cutoff_dh
                          / unit.nanometer)
    yu.addPerParticleParameter("q")

    for i, t in enumerate(tokens):
        ah.addParticle([float(residues.loc[t, "sigmas"]) * unit.nanometer,
                        float(residues.loc[t, "lambdas"])])
        yu.addParticle([yu_eps[i] * unit.nanometer * unit.kilojoule_per_mole])

    for i in range(n - 1):
        hb.addBond(i, i + 1, 0.38 * unit.nanometer,
                   8033.0 * unit.kilojoule_per_mole / unit.nanometer ** 2)
        ah.addExclusion(i, i + 1)
        yu.addExclusion(i, i + 1)

    ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    yu.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setCutoffDistance(cutoff_lj * unit.nanometer)
    yu.setCutoffDistance(cutoff_dh * unit.nanometer)
    hb.setUsesPeriodicBoundaryConditions(True)
    ah.setForceGroup(1)
    yu.setForceGroup(0)

    system.addForce(hb)
    system.addForce(ah)
    system.addForce(yu)
    return system, dict(masses=mw, charges=charges, kappa=float(kappa),
                        lB=float(lB), lj_eps=lj_eps,
                        debye_length_nm=float(1.0 / kappa))


def _platform_props(pname: str, gpu_index: str | None) -> dict:
    if pname in ("CUDA", "OpenCL"):
        props = {"Precision": "mixed"}
        if gpu_index is not None:
            props["DeviceIndex"] = str(gpu_index)
        return props
    return {}


def create_simulation(topology, system, make_integrator, positions,
                      name: str | None, gpu_index: str | None):
    """実際に Context を作れるプラットフォームを順に試して Simulation を返す。

    プラットフォームが一覧に出ていても Context 生成で失敗することがある
    (ドライバより新しい NVRTC による PTX バージョン不一致など)。
    異機種混在のクラスタではノードごとに事情が違うので、
    「一覧にあるか」ではなく「実際にカーネルを作れるか」で判定する。
    CUDA -> OpenCL -> CPU の順にフォールバックする。
    """
    candidates = [name] if name else ["CUDA", "OpenCL", "CPU"]
    errors = []
    for pname in candidates:
        try:
            plat = openmm.Platform.getPlatformByName(pname)
        except Exception:
            errors.append(f"{pname}: このビルドには含まれていません")
            continue
        try:
            sim = app.Simulation(topology, system, make_integrator(), plat,
                                 _platform_props(pname, gpu_index))
            # カーネルのコンパイルまで通ることを確認する
            sim.context.setPositions(positions * unit.nanometer)
            sim.context.getState(getEnergy=True).getPotentialEnergy()
            if errors:
                print(f"  ({' / '.join(errors)} は使えなかったため {pname} に切替)",
                      flush=True)
            return sim, pname
        except Exception as e:
            msg = str(e).split("\n")[0][:120]
            errors.append(f"{pname}: {msg}")
            if "PTX" in str(e):
                print(f"  !! {pname} が PTX バージョン不一致で失敗しました。\n"
                      f"     このノードのドライバに対して NVRTC が新しすぎます。\n"
                      f'     恒久対応: pip install "nvidia-cuda-nvrtc-cu12<12.2" '
                      f"(数字はドライバの CUDA 版 +0.1)", flush=True)
            else:
                print(f"  !! {pname} が使えません: {msg}", flush=True)
    raise RuntimeError("使用可能な OpenMM プラットフォームがありません:\n  "
                       + "\n  ".join(errors))


# ==========================================================================
# メイン
# ==========================================================================
def run(args):
    cfg_all = yaml.safe_load(open(args.config))
    sim_cfg = cfg_all["simulation"]
    if args.system not in cfg_all["systems"]:
        sys.exit(f"system '{args.system}' が config に見つかりません: "
                 f"{list(cfg_all['systems'])}")
    sys_cfg = cfg_all["systems"][args.system]

    # ---- 配列の読み込み ----
    if "sequence" in sys_cfg:
        raw = sys_cfg["sequence"]
    else:
        fa = Path(args.config).parent / sys_cfg["fasta"]
        raw = "".join(l.strip() for l in open(fa) if not l.startswith(">"))
    tokens_in = ff.tokenize(raw)
    mode = sys_cfg.get("mode", "phospho")
    tokens, residues = ff.build_model(
        tokens_in, mode=mode, dlambda=sim_cfg.get("dlambda", ff.DLAMBDA_PHOSPHO))
    n = len(tokens)

    # ---- 実行時パラメータ ----
    dt_ps = sim_cfg["timestep_fs"] * 1e-3
    total_steps = int(round(sim_cfg["total_time_ns"] * 1000 / dt_ps))
    save_every = int(round(sim_cfg["save_interval_ps"] / dt_ps))
    total_steps = (total_steps // save_every) * save_every    # 端数を切る
    box = sim_cfg.get("box_nm", "auto")
    box_nm = float(np.ceil((n - 1) * 0.38 + 4)) if box in (None, "auto") else float(box)

    outdir = Path(args.config).parent / cfg_all.get("outdir", "runs") \
        / f"{args.system}_rep{args.replica}"
    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(sim_cfg.get("seed", 2026)) + 1000 * args.replica + hash(args.system) % 997

    # ---- 情報表示 ----
    info = ff.summarize(tokens, residues, sim_cfg["pH"])
    print("=" * 78)
    print(f" CALVADOS 2 + phospho  |  system={args.system}  replica={args.replica}")
    print("=" * 78)
    print(f"  mode                : {mode}")
    print(f"  N residues          : {n}")
    print(f"  phospho sites (1-based): {ff.phospho_sites(tokens)}")
    print(f"  net charge / NCPR   : {info['net_charge']:+.2f} / {info['NCPR']:+.4f}")
    print(f"  FCR / <lambda>      : {info['FCR']:.4f} / {info['mean_lambda']:.4f}")
    print(f"  T / I / pH          : {sim_cfg['temperature']} K / "
          f"{sim_cfg['ionic_strength']} M / {sim_cfg['pH']}")
    print(f"  box                 : {box_nm} nm")
    print(f"  dt / total          : {sim_cfg['timestep_fs']} fs / "
          f"{sim_cfg['total_time_ns']} ns  ({total_steps:,} steps)")
    print(f"  save every          : {sim_cfg['save_interval_ps']} ps "
          f"({total_steps // save_every:,} frames)")
    print(f"  outdir              : {outdir}")
    print("=" * 78, flush=True)

    # ---- System ----
    system, ffinfo = build_system(tokens, residues, sim_cfg, box_nm)
    print(f"  Debye length        : {ffinfo['debye_length_nm']:.3f} nm")
    print(f"  Bjerrum length      : {ffinfo['lB']:.3f} nm", flush=True)

    top = build_topology(tokens, residues)

    # ---- 初期構造 ----
    import mdtraj as md
    ckpt = outdir / "checkpoint.chk"
    dcd_path = outdir / "traj.dcd"
    restart = ckpt.exists() and dcd_path.exists() and not args.fresh

    # レプリカごとに初期構造の広がりを変える (rep0 = 最もコンパクトな螺旋)
    if args.replica == 0:
        pos = spiral_positions(n)
    else:
        stiffness = [0.0, 0.0, 0.45, 0.70, 0.85][min(args.replica, 4)]
        pos = random_walk_positions(n, seed=seed, stiffness=stiffness)
        # 伸びすぎてボックスからはみ出さないように縮める
        span = pos.max(axis=0) - pos.min(axis=0)
        if span.max() > 0.85 * box_nm:
            pos *= 0.85 * box_nm / span.max()
    pos = pos - pos.mean(axis=0) + box_nm / 2.0
    pdb_top = outdir / "top.pdb"
    md.Trajectory(pos[None, :, :], top, 0,
                  [box_nm] * 3, [90, 90, 90]).save_pdb(str(pdb_top),
                                                       force_overwrite=True)

    # ---- Integrator / Simulation ----
    # Context 生成に失敗した Integrator は再利用できないので、毎回作り直せるよう
    # ファクトリにしておく (プラットフォームのフォールバック用)
    def make_integrator():
        integ = openmm.LangevinMiddleIntegrator(
            sim_cfg["temperature"] * unit.kelvin,
            sim_cfg["friction_per_ps"] / unit.picosecond,
            dt_ps * unit.picosecond)
        integ.setRandomNumberSeed(seed % (2 ** 31 - 1))
        return integ

    gpu_index = args.gpu
    if gpu_index is None and os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_index = "0"       # CUDA_VISIBLE_DEVICES で既に絞られている

    omm_top = app.PDBFile(str(pdb_top)).topology
    simulation, pname = create_simulation(
        omm_top, system, make_integrator, pos, args.platform, gpu_index)
    print(f"  platform            : {pname} "
          f"{'(device ' + str(gpu_index) + ')' if gpu_index is not None else ''}",
          flush=True)

    start_step = 0
    if restart:
        try:
            simulation.loadCheckpoint(str(ckpt))
            start_step = simulation.currentStep
            print(f"  --> チェックポイントから再開: step {start_step:,} "
                  f"({start_step * dt_ps / 1000:.1f} ns)", flush=True)
        except Exception as e:
            print(f"  !! チェックポイント読み込み失敗 ({e}) -> 最初から", flush=True)
            restart = False

    if not restart:
        simulation.context.setPositions(pos * unit.nanometer)
        print("  エネルギー最小化 ...", flush=True)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(
            sim_cfg["temperature"] * unit.kelvin, seed % (2 ** 31 - 1))
        simulation.currentStep = 0
        if dcd_path.exists():
            shutil.move(str(dcd_path), str(dcd_path) + ".bak")

    if start_step >= total_steps:
        print("  既に完了しています。", flush=True)
        return

    # ---- Reporters ----
    # 順番が重要: DCD -> checkpoint とすることで、再開時にフレームが重複しない
    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), save_every,
                        enforcePeriodicBox=False, append=restart))
    simulation.reporters.append(
        app.StateDataReporter(str(outdir / "state.log"), save_every,
                              step=True, time=True, potentialEnergy=True,
                              kineticEnergy=True, temperature=True,
                              speed=True, elapsedTime=True, separator="\t",
                              append=restart))
    prog_every = max(save_every, total_steps // int(args.progress_points))
    prog_every = (prog_every // save_every) * save_every or save_every
    simulation.reporters.append(
        ProgressReporter(outdir / "status.json", total_steps, prog_every, dt_ps,
                         ffinfo["masses"],
                         label=f"{args.system}/rep{args.replica}",
                         start_step=start_step))
    simulation.reporters.append(
        app.CheckpointReporter(str(ckpt), save_every))

    # ---- メタ情報を保存 ----
    meta = dict(
        system=args.system, replica=args.replica, mode=mode,
        sequence_tokens=tokens, n_residues=n,
        phospho_sites=ff.phospho_sites(tokens),
        box_nm=box_nm, seed=seed, platform=pname,
        total_steps=total_steps, save_every=save_every,
        dt_ps=dt_ps, sim_config=dict(sim_cfg),
        masses=ffinfo["masses"].tolist(),
        charges=ffinfo["charges"].tolist(),
        debye_length_nm=ffinfo["debye_length_nm"],
        composition=info,
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    residues.to_csv(outdir / "residues_used.csv")

    # ---- 本番 ----
    max_hours = args.max_hours if args.max_hours is not None \
        else sim_cfg.get("max_wall_hours", 0) or 0
    if max_hours:
        print(f"  壁時計時間の上限     : {max_hours} h "
              f"(超えたらチェックポイントを保存して exit {EXIT_REQUEUE})", flush=True)

    t0 = time.time()
    while simulation.currentStep < total_steps:
        simulation.step(min(save_every, total_steps - simulation.currentStep))
        if max_hours and (time.time() - t0) / 3600.0 >= max_hours:
            simulation.saveCheckpoint(str(ckpt))
            done_ns = simulation.currentStep * dt_ps / 1000
            print(f"\n壁時計時間の上限 ({max_hours} h) に達しました。"
                  f"{done_ns:.1f}/{sim_cfg['total_time_ns']} ns まで完了。"
                  f"\nチェックポイントを保存して終了します "
                  f"(exit {EXIT_REQUEUE})。再実行すれば続きから走ります。", flush=True)
            sys.exit(EXIT_REQUEUE)
    dt_tot = time.time() - t0
    print(f"\n完了: {args.system}/rep{args.replica}  "
          f"{dt_tot / 3600:.2f} h  ({(total_steps - start_step) * dt_ps / 1000 / (dt_tot / 86400):.0f} ns/day)",
          flush=True)

    # 最終状態を保存
    simulation.saveCheckpoint(str(ckpt))
    state = simulation.context.getState(getPositions=True)
    with open(outdir / "final.pdb", "w") as fh:
        app.PDBFile.writeFile(omm_top, state.getPositions(), fh)

    st = json.loads((outdir / "status.json").read_text())
    st["finished"] = True
    (outdir / "status.json").write_text(json.dumps(st, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--system", required=True, help="config.yaml の systems のキー")
    p.add_argument("--replica", type=int, default=0)
    p.add_argument("--gpu", default=None,
                   help="CUDA の DeviceIndex。未指定なら CUDA_VISIBLE_DEVICES に従う")
    p.add_argument("--platform", default=None,
                   choices=["CUDA", "OpenCL", "CPU", "Reference"])
    p.add_argument("--fresh", action="store_true",
                   help="チェックポイントを無視して最初から走らせる")
    p.add_argument("--max-hours", type=float, default=None,
                   help="この壁時計時間を超えたらチェックポイントを保存して "
                        f"exit {EXIT_REQUEUE} する (既定は config の max_wall_hours)")
    p.add_argument("--progress-points", type=int, default=500,
                   help="進捗を何回出力するか (デフォルト 500 = 0.2%% 刻み)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
