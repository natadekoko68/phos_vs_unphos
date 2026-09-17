#!/usr/bin/env python
"""
benchmark.py
============
本番の 1 us ランを流す前に、環境と速度を確認する。

  * OpenMM が CUDA を見つけられているか
  * GPU が何枚あるか
  * この系で実際に何 ns/day 出るか -> 1 us にかかる時間の見積もり

    python scripts/benchmark.py                     # 既定 (phos, 20000 steps)
    python scripts/benchmark.py --system unphos --steps 50000
    python scripts/benchmark.py --all-gpus          # 全 GPU を 1 枚ずつ測る
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
import openmm
from openmm import app, unit

import calvados_ff as ff
from simulate import (build_system, build_topology, spiral_positions,
                      create_simulation)


def explain_openmm_error(e: Exception):
    """よくある OpenMM/CUDA のエラーに対処法を出す。"""
    msg = str(e)
    if "PTX" in msg or "222" in msg:
        print("""
  --------------------------------------------------------------------
  原因: NVRTC (CUDA の実行時コンパイラ) がドライバより新しすぎます。

  OpenMM は CUDA カーネルを NVRTC で実行時コンパイルしますが、pip は
  依存解決で最新の nvidia-cuda-nvrtc-cu12 を入れてしまうため、生成された
  PTX をドライバが読めずに落ちます。

  対処: ドライバが対応する CUDA 版以下に固定してください。
        nvidia-smi の右上の "CUDA Version:" を見て、その次のマイナー版未満に:

      pip install "nvidia-cuda-nvrtc-cu12<12.2"    # CUDA 12.1 のドライバなら
      pip install "nvidia-cuda-nvrtc-cu12<12.5"    # CUDA 12.4 のドライバなら

  それでも直らないときは OpenCL でも走ります (T4 なら 2-3 割遅い程度):

      python simulate.py --platform OpenCL ...
  --------------------------------------------------------------------""")
    elif "no CUDA-capable device" in msg or "CUDA_ERROR_NO_DEVICE" in msg:
        print("  -> GPU が見えていません。CUDA_VISIBLE_DEVICES を確認してください。")
    elif "out of memory" in msg.lower():
        print("  -> GPU メモリ不足。ボックスサイズか他ジョブの状況を確認してください。")


def list_platforms():
    print("OpenMM version :", openmm.version.version)
    names = [openmm.Platform.getPlatform(i).getName()
             for i in range(openmm.Platform.getNumPlatforms())]
    print("Platforms      :", ", ".join(names))
    return names


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        gpus = [l for l in out.splitlines() if l]
        print(f"GPU            : {len(gpus)} 枚")
        for g in gpus:
            print("   ", g)
        return len(gpus)
    except Exception:
        print("GPU            : nvidia-smi が見つかりません")
        return 0


def run_benchmark(cfg_path, system, steps, gpu, platform_name):
    cfg_all = yaml.safe_load(open(cfg_path))
    sim_cfg = cfg_all["simulation"]
    sys_cfg = cfg_all["systems"][system]

    if "sequence" in sys_cfg:
        raw = sys_cfg["sequence"]
    else:
        raw = "".join(l.strip() for l in
                      open(Path(cfg_path).parent / sys_cfg["fasta"])
                      if not l.startswith(">"))
    tokens, residues = ff.build_model(ff.tokenize(raw),
                                      mode=sys_cfg.get("mode", "phospho"),
                                      dlambda=sim_cfg.get("dlambda", -0.37))
    n = len(tokens)
    box = sim_cfg.get("box_nm", "auto")
    box_nm = float(np.ceil((n - 1) * 0.38 + 4)) if box in (None, "auto") else float(box)

    system_omm, info = build_system(tokens, residues, sim_cfg, box_nm)
    top = build_topology(tokens, residues)
    pos = spiral_positions(n) + box_nm / 2.0

    import mdtraj as md
    tmp = Path("/tmp/_bench_top.pdb")
    md.Trajectory(pos[None], top, 0, [box_nm] * 3, [90, 90, 90]).save_pdb(
        str(tmp), force_overwrite=True)

    dt_ps = sim_cfg["timestep_fs"] * 1e-3

    sim, pname = create_simulation(app.PDBFile(str(tmp)).topology, system_omm,
                                   lambda: openmm.LangevinMiddleIntegrator(
                                       sim_cfg["temperature"] * unit.kelvin,
                                       sim_cfg["friction_per_ps"] / unit.picosecond,
                                       dt_ps * unit.picosecond),
                                   pos, platform_name, gpu)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(sim_cfg["temperature"] * unit.kelvin)

    sim.step(min(2000, steps // 5))          # warm-up (JIT コンパイル分を除く)
    t0 = time.time()
    sim.step(steps)
    dt = time.time() - t0

    sps = steps / dt
    ns_day = sps * dt_ps * 1e-3 * 86400
    total_ns = sim_cfg["total_time_ns"]
    hours = total_ns / ns_day * 24
    return dict(platform=pname, gpu=gpu, n_residues=n, box_nm=box_nm,
                steps_per_s=sps, ns_per_day=ns_day,
                hours_for_target=hours, target_ns=total_ns)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    p.add_argument("--config", default=str(root / "config.yaml"))
    p.add_argument("--system", default=None, help="既定は config の最初の系")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--platform", default=None,
                   choices=["CUDA", "OpenCL", "CPU", "Reference"])
    p.add_argument("--all-gpus", action="store_true")
    a = p.parse_args()

    print("=" * 72)
    list_platforms()
    ngpu = gpu_info()
    print("=" * 72)

    cfg = yaml.safe_load(open(a.config))
    system = a.system or list(cfg["systems"])[0]
    devices = [str(i) for i in range(ngpu)] if (a.all_gpus and ngpu) else [None]

    results = []
    for dev in devices:
        print(f"\nベンチマーク中 (system={system}, "
              f"device={dev if dev is not None else 'auto'}, "
              f"{a.steps:,} steps) ...", flush=True)
        try:
            r = run_benchmark(a.config, system, a.steps, dev, a.platform)
        except openmm.OpenMMException as e:
            print(f"  失敗: {e}")
            explain_openmm_error(e)
            continue
        results.append(r)
        print(f"  platform  : {r['platform']}")
        print(f"  速度      : {r['steps_per_s']:,.0f} steps/s = "
              f"{r['ns_per_day']:,.0f} ns/day")
        print(f"  {r['target_ns']:g} ns 1 本あたり: {r['hours_for_target']:.2f} 時間")

    if results:
        best = max(results, key=lambda r: r["ns_per_day"])
        nrep = cfg["simulation"].get("n_replicas", 4)
        njobs = len(cfg["systems"]) * nrep
        print("\n" + "=" * 72)
        print(f"系 {len(cfg['systems'])} × レプリカ {nrep} = {njobs} ジョブ")
        if ngpu:
            waves = int(np.ceil(njobs / ngpu))
            print(f"GPU {ngpu} 枚 -> {waves} 巡  "
                  f"= 実時間 約 {waves * best['hours_for_target']:.1f} 時間")
        print("=" * 72)


if __name__ == "__main__":
    main()
