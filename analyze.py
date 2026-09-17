#!/usr/bin/env python
"""
analyze.py
==========
CALVADOS シミュレーションの解析:

  * 慣性半径 Rg の時系列と分布 (KDE + ヒストグラム)
  * ブロック平均法による <Rg> の統計誤差
  * 末端間距離 Ree、内部スケーリング指数 nu
  * 系間の比較 (非リン酸化 vs リン酸化) と論文の指標 dRg/Rg^ph
  * 最もコンパクトな構造 / 最も伸びた構造の抽出 (PDB)
  * 残基間距離マップとその差分

使い方
------
    python analyze.py --config config.yaml
    python analyze.py --config config.yaml --systems unphos phos --outdir analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import mdtraj as md
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

import calvados_ff as ff

PALETTE = {"unphos": "#4C72B0", "phos": "#C44E52",
           "phos_charge": "#DD8452", "phos_mimetic": "#55A868"}

# --- 日本語フォントがあれば日本語ラベル、無ければ英語ラベルにフォールバック ---
CJK_FONTS = ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAPGothic",
             "TakaoGothic", "VL PGothic", "Hiragino Sans", "Yu Gothic",
             "MS Gothic", "Source Han Sans"]

L_JA = dict(
    dist_title="慣性半径の分布  (点線 = 抽出した最小/最大 $R_g$)",
    prob="確率密度", cdf_y="累積確率", cdf_title="累積分布 (CDF)",
    ts_x="累積シミュレーション時間 (ns, レプリカを連結)",
    ts_title="$R_g$ の時系列 (灰線 = レプリカの区切り)",
    ree_title="末端間距離", scal_title="内部スケーリング",
    dmap_x="残基 i", dmap_y="残基 j",
    summary="【まとめ】", nframes="フレーム",
    change="リン酸化による変化", pos_note="(正 = リン酸化で伸長)",
    suptitle="CALVADOS 2 + リン酸化モデルによる IDR コンフォメーション解析",
    single_title="リン酸化による $R_g$ 分布の変化",
    pca_landscape="PCA ランドスケープ (残基間距離ベース)",
    pca_var="寄与率",
    pca_loading="PC1 の loading (どの接触が PC1 を決めているか)",
    pca_suptitle="コンフォメーションアンサンブルの主成分分析 (距離行列 PCA)",
)
L_EN = dict(
    dist_title="Radius of gyration distribution (dotted = extracted min/max $R_g$)",
    prob="probability density", cdf_y="cumulative probability", cdf_title="CDF",
    ts_x="cumulative simulation time (ns, replicas concatenated)",
    ts_title="$R_g$ time series (grey lines = replica boundaries)",
    ree_title="end-to-end distance", scal_title="internal scaling",
    dmap_x="residue i", dmap_y="residue j",
    summary="[ summary ]", nframes="frames",
    change="effect of phosphorylation", pos_note="(positive = expansion)",
    suptitle="CALVADOS 2 + phosphorylation model: IDR conformational analysis",
    single_title="Effect of phosphorylation on the $R_g$ distribution",
)
LBL = L_EN          # setup_fonts() で切り替わる


def setup_fonts():
    """日本語フォントを探して matplotlib に設定する。無ければ英語ラベルにする。"""
    global LBL
    # 1) japanize-matplotlib が入っていればそれに任せる (pip install japanize-matplotlib)
    try:
        import japanize_matplotlib  # noqa: F401
        LBL = L_JA
        print("  日本語フォント: japanize-matplotlib")
        plt.rcParams["axes.unicode_minus"] = False
        return
    except Exception:
        pass

    # 2) システムにある CJK フォントを探す
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fam in CJK_FONTS:
        if any(fam in a for a in available):
            plt.rcParams["font.family"] = [fam, "DejaVu Sans"]
            LBL = L_JA
            print(f"  日本語フォント: {fam}")
            break
    else:
        # 3) 無ければ英語ラベルにフォールバック (図は問題なく出ます)
        LBL = L_EN
        print("  (日本語フォントが無いため図は英語ラベルにします。日本語にするには "
              "'pip install japanize-matplotlib' か "
              "'sudo apt install fonts-noto-cjk')")
    plt.rcParams["axes.unicode_minus"] = False


# ==========================================================================
# 基本量
# ==========================================================================

def unwrap_chain(traj: md.Trajectory) -> md.Trajectory:
    """周期境界で切れた鎖を最小イメージ規約でつなぎ直す。

    DCD は enforcePeriodicBox=False で書いているので通常は不要だが、
    念のため常に適用しておく (すでに連続なら何も変わらない)。
    """
    if traj.unitcell_lengths is None:
        return traj
    L = np.asarray(traj.unitcell_lengths)[:, None, :]      # (nf, 1, 3)
    x = np.array(traj.xyz, dtype=np.float32)
    d = np.diff(x, axis=1)
    d -= L * np.round(d / L)
    x[:, 1:, :] = x[:, :1, :] + np.cumsum(d, axis=1)
    traj.xyz = x
    return traj


def compute_rg(traj: md.Trajectory, masses: np.ndarray) -> np.ndarray:
    """質量重み付き慣性半径 [nm]。"""
    m = masses[None, :, None]
    com = (traj.xyz * m).sum(axis=1) / masses.sum()
    d2 = ((traj.xyz - com[:, None, :]) ** 2).sum(axis=2)
    return np.sqrt((d2 * masses[None, :]).sum(axis=1) / masses.sum())


def compute_ree(traj: md.Trajectory) -> np.ndarray:
    """末端間距離 [nm]。"""
    return np.linalg.norm(traj.xyz[:, -1, :] - traj.xyz[:, 0, :], axis=1)


def block_error(x: np.ndarray, min_blocks: int = 10):
    """ブロック平均法で相関を考慮した平均の標準誤差を推定する。

    ブロック長を変えながら SEM を計算し、プラトーに達した値を採用する
    (Flyvbjerg & Petersen 1989 の考え方)。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    sizes, sems = [], []
    bs = 1
    while n // bs >= min_blocks:
        nb = n // bs
        means = x[: nb * bs].reshape(nb, bs).mean(axis=1)
        sizes.append(bs)
        sems.append(means.std(ddof=1) / np.sqrt(nb))
        bs = max(bs + 1, int(bs * 1.3))
    sizes, sems = np.array(sizes), np.array(sems)
    # SEM が単調増加から頭打ちになる点 = 最大値付近を採用（最後の 1/3 の中央値）
    tail = sems[max(1, len(sems) * 2 // 3):]
    sem = float(np.median(tail)) if len(tail) else float(sems[-1])
    # 統計的非効率から有効サンプル数を出す
    n_eff = (x.std(ddof=1) / sem) ** 2 if sem > 0 else float(n)
    return float(x.mean()), sem, float(n_eff), sizes, sems


def internal_scaling(traj: md.Trajectory, stride: int = 1):
    """|i-j| に対する平均二乗距離から Flory 指数 nu をフィットする。"""
    t = traj[::stride]
    pairs = t.top.select_pairs("all", "all")
    d = md.compute_distances(t, pairs)
    sep = np.abs(pairs[:, 1] - pairs[:, 0])
    ij = np.arange(2, t.n_atoms)
    dij = np.array([np.sqrt(np.mean(d[:, sep == s] ** 2)) for s in ij])
    f = lambda x, R0, nu: R0 * np.power(x, nu)
    mask = ij > 5
    popt, pcov = curve_fit(f, ij[mask], dij[mask], p0=[0.55, 0.5])
    return ij, dij, popt[0], popt[1], float(np.sqrt(pcov[1, 1]))


def mean_distance_map(traj: md.Trajectory, stride: int = 5) -> np.ndarray:
    t = traj[::stride]
    pairs = t.top.select_pairs("all", "all")
    d = md.compute_distances(t, pairs).mean(axis=0)
    n = t.n_atoms
    M = np.zeros((n, n))
    M[pairs[:, 0], pairs[:, 1]] = d
    M += M.T
    return M


# ==========================================================================
# 主成分分析 (PCA)
# ==========================================================================
#
# 折り畳みタンパク質と違って IDR には単一の参照構造が無いため、Cartesian
# 座標を重ね合わせて PCA する方法 (通常の "Cartesian PCA") はあまり意味を
# 持たない。代わりに、CALVADOS/IDR アンサンブル解析で標準的な
# "距離行列 PCA (distance-matrix PCA / dPCA-like)" を使う:
#
#   1. 各フレームについて、残基対 (i, j) 間の距離をすべて計算し、
#      1 本のベクトルに平坦化する (= 特徴量)
#   2. 非リン酸化・リン酸化の両アンサンブルのフレームをまとめて標準的な
#      PCA にかけ、共通の主成分空間を作る
#   3. 各系のフレームをその空間に射影して比較する
#
# 392 残基だと残基対は 392*391/2 ≈ 76,600 個あり、全フレーム分を密行列で
# 保持すると数十 GB になり得るため、
#   * 残基を間引く (pca.residue_stride)
#   * フィッティングに使うフレーム数を間引く (pca.max_frames_per_system)
# の 2 段階でサイズを抑える。射影 (transform) は間引かず全フレームに対して
# 行うので、Rg などと同じ分だけ統計量が得られる。
# ==========================================================================
def pca_pairs(n_atoms: int, residue_stride: int) -> np.ndarray:
    """PCA の特徴量に使う残基対 (間引きあり) を返す。"""
    idx = np.arange(0, n_atoms, max(1, residue_stride))
    ii, jj = np.triu_indices(len(idx), k=1)
    return np.stack([idx[ii], idx[jj]], axis=1)


def pairwise_distance_features(traj: md.Trajectory, pairs: np.ndarray,
                               frame_idx: np.ndarray | None = None) -> np.ndarray:
    """指定した残基対について距離を計算し (n_frames, n_pairs) で返す。"""
    t = traj if frame_idx is None else traj[frame_idx]
    return md.compute_distances(t, pairs).astype(np.float32)


def run_pca(results: dict, trajs: dict, acfg: dict, outdir: Path):
    """全系のフレームをまとめて PCA を行い、各系を射影する。

    Parameters
    ----------
    results : analyze() 内で組み立てた系ごとの結果 dict (rg などを含む)
    trajs   : {system: mdtraj.Trajectory} (load_system が返した連結済み軌跡)
    """
    from sklearn.decomposition import PCA

    systems = list(trajs)
    n_atoms = trajs[systems[0]].n_atoms
    res_stride = int(acfg.get("pca_residue_stride", 3))
    max_fit = int(acfg.get("pca_max_frames_per_system", 4000))
    n_comp = int(acfg.get("pca_n_components", 5))

    pairs = pca_pairs(n_atoms, res_stride)
    print(f"\n=== PCA ===")
    print(f"  特徴量: 残基 {res_stride} 個おきに間引き -> "
          f"{len(np.unique(pairs))} 残基, {len(pairs)} 残基対")

    # ---- フィッティング用にフレームを間引いて集める ----
    fit_blocks, fit_labels = [], []
    rng = np.random.default_rng(0)
    for s in systems:
        t = trajs[s]
        n = t.n_frames
        if n > max_fit:
            idx = np.sort(rng.choice(n, size=max_fit, replace=False))
        else:
            idx = np.arange(n)
        fit_blocks.append(pairwise_distance_features(t, pairs, idx))
        fit_labels += [s] * len(idx)
        print(f"  {s}: フィッティングに {len(idx)}/{n} フレーム使用")

    X_fit = np.concatenate(fit_blocks, axis=0)
    print(f"  PCA 入力行列: {X_fit.shape} "
          f"({X_fit.nbytes / 1e6:.0f} MB)")

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)
    pca.fit(X_fit)
    print(f"  寄与率 (PC1..PC{n_comp}): "
          + ", ".join(f"{v * 100:.1f}%" for v in pca.explained_variance_ratio_))

    # ---- 各系の全フレームを射影 (統計量を落とさないよう間引かない) ----
    proj, rg_all = {}, {}
    for s in systems:
        t = trajs[s]
        X = pairwise_distance_features(t, pairs)
        proj[s] = pca.transform(X)
        rg_all[s] = results[s]["rg"]
        df = pd.DataFrame(proj[s], columns=[f"PC{i+1}" for i in range(n_comp)])
        df["Rg_nm"] = rg_all[s]
        df.to_csv(outdir / f"pca_projection_{s}.csv", index=False)

    # PC1 と Rg の相関 (IDR では PC1 が概ね広がり具合に対応することが多い)
    corr = {s: float(np.corrcoef(proj[s][:, 0], rg_all[s])[0, 1]) for s in systems}

    # ---- PC1 に沿った極端構造も保存 (Rg とは独立な観点になりうる) ----
    struct_dir = outdir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    n_each = int(acfg.get("n_extreme_structures", 5))
    for s in systems:
        t = trajs[s]
        pc1 = proj[s][:, 0]
        order = np.argsort(pc1)
        ref = t[int(order[0])]
        for tag, idxs in [("PC1_low", order[:n_each]),
                          ("PC1_high", order[::-1][:n_each])]:
            for rank, i in enumerate(idxs, start=1):
                fr = t[int(i)]
                fr.superpose(ref)
                fr.save_pdb(str(struct_dir /
                                f"{s}_{tag}{rank:02d}_PC1{pc1[int(i)]:+.2f}.pdb"))

    return dict(pca=pca, pairs=pairs, proj=proj, corr=corr,
               n_comp=n_comp, res_stride=res_stride)


def plot_pca(pca_res: dict, results: dict, outdir: Path):
    """PCA landscape (PC1-PC2)、寄与率、PC1 の loading map を 1 枚にまとめる。"""
    pca, proj, corr = pca_res["pca"], pca_res["proj"], pca_res["corr"]
    systems = list(proj)

    fig = plt.figure(figsize=(13, 4.6), dpi=150)
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- (0) PC1-PC2 の散布図 + KDE 等高線 ---
    ax = fig.add_subplot(gs[0, 0])
    for s in systems:
        c = PALETTE.get(s)
        p = proj[s]
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.15, color=c, linewidths=0,
                  rasterized=True)
        try:
            kde = gaussian_kde(p[:, :2].T)
            xg = np.linspace(p[:, 0].min(), p[:, 0].max(), 80)
            yg = np.linspace(p[:, 1].min(), p[:, 1].max(), 80)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contour(XX, YY, ZZ, levels=5, colors=[c], linewidths=1.0)
        except Exception:
            pass
        ax.scatter([], [], color=c, label=results[s]["label"])   # legend 用
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(LBL.get("pca_landscape", "PCA landscape (distance-based)"))
    ax.legend(fontsize=8, markerscale=3)

    # --- (1) 寄与率 ---
    ax = fig.add_subplot(gs[0, 1])
    n = pca_res["n_comp"]
    ax.bar(np.arange(1, n + 1), pca.explained_variance_ratio_ * 100,
          color="0.5")
    ax.plot(np.arange(1, n + 1),
            np.cumsum(pca.explained_variance_ratio_) * 100,
            "o-", color="k", ms=4, label="cumulative")
    ax.set_xlabel("PC index"); ax.set_ylabel("explained variance (%)")
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_title(LBL.get("pca_var", "explained variance"))
    ax.legend(fontsize=8)
    txt = "\n".join(f"{s}: corr(PC1,Rg) = {corr[s]:+.2f}" for s in systems)
    ax.text(0.98, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, family="monospace")

    # --- (2) PC1 loading map (残基対ごとの寄与を N x N で可視化) ---
    ax = fig.add_subplot(gs[0, 2])
    pairs = pca_res["pairs"]
    n_res = int(pairs.max()) + 1
    load = np.zeros((n_res, n_res))
    load[pairs[:, 0], pairs[:, 1]] = pca.components_[0]
    load += load.T
    v = np.nanpercentile(np.abs(load), 99) or 1.0
    im = ax.imshow(load, cmap="PuOr_r", vmin=-v, vmax=v, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, label="PC1 loading")
    ax.set_xlabel("residue i"); ax.set_ylabel("residue j")
    ax.set_title(LBL.get("pca_loading", "PC1 loading (which contacts drive PC1)"))

    fig.suptitle(LBL.get("pca_suptitle",
                         "Distance-matrix PCA of the conformational ensemble"),
                fontsize=12, y=1.03)
    fig.savefig(outdir / "pca.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# 読み込み
# ==========================================================================
def load_system(root: Path, system: str, equil_ns: float):
    """1 つの系の全レプリカを読み込み、平衡化部分を捨てて連結する。"""
    dirs = sorted(root.glob(f"{system}_rep*"),
                  key=lambda p: int(p.name.split("rep")[-1]))
    if not dirs:
        raise FileNotFoundError(f"{root}/{system}_rep* が見つかりません")

    trajs, per_rep, meta = [], [], None
    for d in dirs:
        dcd, top = d / "traj.dcd", d / "top.pdb"
        if not dcd.exists():
            print(f"  [skip] {d.name}: traj.dcd がありません")
            continue
        meta = json.loads((d / "meta.json").read_text())
        dt_frame_ns = meta["save_every"] * meta["dt_ps"] / 1000.0
        t = md.load_dcd(str(dcd), top=str(top))
        n_equil = int(round(equil_ns / dt_frame_ns))
        if t.n_frames <= n_equil:
            print(f"  [skip] {d.name}: フレーム数 {t.n_frames} <= 平衡化 {n_equil}")
            continue
        t = t[n_equil:]
        t = unwrap_chain(t)          # PBC で鎖が切れていないことを保証
        t.center_coordinates()
        trajs.append(t)
        per_rep.append(dict(replica=int(d.name.split("rep")[-1]),
                            n_frames=t.n_frames, dt_frame_ns=dt_frame_ns,
                            dir=str(d)))
        print(f"  [ok] {d.name}: {t.n_frames} frames "
              f"({t.n_frames * dt_frame_ns:.4g} ns, {dt_frame_ns:g} ns/frame)")

    if not trajs:
        raise RuntimeError(f"{system}: 使えるトラジェクトリがありません")
    traj = trajs[0] if len(trajs) == 1 else md.join(trajs, check_topology=False)
    return traj, trajs, per_rep, meta


# ==========================================================================
# 分子内相互作用の解析
# ==========================================================================
#
# 「距離」ではなく実際の相互作用そのものに踏み込むための解析:
#
#   1. 形状記述子 (慣性楕円体) — 相互作用の結果として鎖がどんな形になるか
#   2. 接触確率マップ — どの残基対がどれくらいの頻度で触れ合っているか
#   3. 電荷タイプ別の接触エンリッチメント — pSer が K/R と選択的に
#      相互作用しているか (静電引力の直接証拠)
#   4. 相互作用エネルギーの分解 (AH: 疎水性/排除体積, DH: 塩遮蔽静電) —
#      simulate.py と全く同じ式で厳密計算 (calvados_ff.py 内で検証済み)
#
# 力場パラメータは各系の meta.json (sequence_tokens, sim_config) から
# 正確に再構築するので、実際にそのシミュレーションで使われた値と
# 完全に一致する。
# ==========================================================================
def gyration_tensor_shape(traj: md.Trajectory, masses: np.ndarray) -> dict:
    """慣性楕円体の形状記述子 (Theodorou & Suter 1985 の定義)。

    固有値 λ1 >= λ2 >= λ3 (Rg^2 = λ1+λ2+λ3) から:
      asphericity b      = λ1 - 0.5*(λ2+λ3)   (0 = 完全な球、大きいほど非球状)
      acylindricity c     = λ2 - λ3            (0 = 円柱対称)
      shape anisotropy κ² = (b² + 0.75 c²) / Rg^4  (0 = 球、1 = 直線状)
    """
    m = masses[None, :, None]
    com = (traj.xyz * m).sum(axis=1) / masses.sum()
    d = traj.xyz - com[:, None, :]
    # 質量重み付き回転半径テンソル (n_frames, 3, 3)
    S = np.einsum("nia,nib,i->nab", d, d, masses) / masses.sum()
    eig = np.linalg.eigvalsh(S)                 # 昇順 (λ3<=λ2<=λ1)
    l3, l2, l1 = eig[:, 0], eig[:, 1], eig[:, 2]
    rg2 = l1 + l2 + l3
    b = l1 - 0.5 * (l2 + l3)
    c = l2 - l3
    kappa2 = (b ** 2 + 0.75 * c ** 2) / np.maximum(rg2, 1e-12) ** 2
    return dict(asphericity=b, acylindricity=c, kappa2=kappa2,
               rg2=rg2, eigenvalues=eig)


def _residue_ff_arrays(meta: dict):
    """meta.json から、そのランで実際に使われた力場パラメータを再構築する。"""
    tokens = meta["sequence_tokens"]
    sc = meta["sim_config"]
    residues = ff.load_residues(dlambda=sc.get("dlambda", ff.DLAMBDA_PHOSPHO))
    r = ff.apply_ph(residues, sc["pH"])
    sigmas = np.array([float(r.loc[t, "sigmas"]) for t in tokens])
    lambdas = np.array([float(r.loc[t, "lambdas"]) for t in tokens])
    yu_eps, kappa, charges, lB = ff.debye_and_charges(
        tokens, residues, sc["temperature"], sc["ionic_strength"], sc["pH"])
    categories = ff.classify_residues(tokens)
    return dict(tokens=tokens, sigmas=sigmas, lambdas=lambdas,
               yu_eps=yu_eps, kappa=kappa, charges=charges,
               categories=categories,
               eps=sc.get("eps_factor", 0.2) * 4.184,
               rc_lj=sc.get("cutoff_lj_nm", 2.0),
               rc_dh=sc.get("cutoff_dh_nm", 4.0))


def interaction_analysis(traj: md.Trajectory, meta: dict, stride: int = 5,
                         contact_cutoff_factor: float = 1.5) -> dict:
    """接触確率・エネルギー分解・電荷タイプ別エンリッチメントをまとめて計算する。

    重い O(N^2) 計算 (距離行列) は計算量を抑えるため stride 間引きしたフレーム
    に対して行う (mean_distance_map / internal_scaling と同じ考え方)。
    """
    fp = _residue_ff_arrays(meta)
    n = len(fp["tokens"])
    t = traj[::stride]

    # 結合している隣接残基 (i, i+1) は力場側でも除外されているペアなので、
    # 「相互作用」の対象からも除く
    all_pairs = t.top.select_pairs("all", "all")
    keep = np.abs(all_pairs[:, 1] - all_pairs[:, 0]) >= 2
    pairs = all_pairs[keep]
    ri, rj = pairs[:, 0], pairs[:, 1]

    dist = md.compute_distances(t, pairs).astype(np.float64)   # (n_frames, n_pairs)

    sigma_ij = 0.5 * (fp["sigmas"][ri] + fp["sigmas"][rj])
    lambda_ij = 0.5 * (fp["lambdas"][ri] + fp["lambdas"][rj])
    e_ah = ff.ashbaugh_hatch_energy(dist, sigma_ij, lambda_ij, fp["eps"], fp["rc_lj"])
    e_dh = ff.yukawa_energy(dist, fp["yu_eps"][ri], fp["yu_eps"][rj],
                            fp["kappa"], fp["rc_dh"])

    # --- 接触確率 (幾何学的定義: r < contact_cutoff_factor * sigma_ij) ---
    cutoff = contact_cutoff_factor * sigma_ij
    contact = dist < cutoff[None, :]
    contact_prob = contact.mean(axis=0)                         # (n_pairs,)
    n_contacts_frame = contact.sum(axis=1)                       # (n_frames,)

    # --- 残基ごとのエネルギー寄与 (ペアのエネルギーを両端に半分ずつ配分) ---
    res_ah = np.zeros(n); res_dh = np.zeros(n)
    mean_e_ah, mean_e_dh = e_ah.mean(axis=0), e_dh.mean(axis=0)
    np.add.at(res_ah, ri, 0.5 * mean_e_ah)
    np.add.at(res_ah, rj, 0.5 * mean_e_ah)
    np.add.at(res_dh, ri, 0.5 * mean_e_dh)
    np.add.at(res_dh, rj, 0.5 * mean_e_dh)

    # --- 電荷タイプ別の接触エンリッチメント (観測 / 期待) ---
    #
    # 単純に「全ペアの平均」を期待値にすると、配列上で近い残基同士
    # (|i-j| が小さい) は鎖の連結性だけで接触しやすいため、たまたま近くに
    # 集まっている電荷タイプ (今回で言えば密集した pSer クラスター) の
    # エンリッチメントを過大評価してしまう。
    # そこで、同じ配列間隔 |i-j| を持つ「全ペアの平均接触確率」を期待値と
    # する sequence-separation 補正版も計算する。こちらのほうが
    # 「鎖がたまたま近いから」ではなく「電荷的に引き合っているから」接触
    # しているかを見るのに適している。
    sep = np.abs(rj - ri)
    uniq_sep, inv = np.unique(sep, return_inverse=True)
    sep_sum = np.zeros(len(uniq_sep)); sep_cnt = np.zeros(len(uniq_sep))
    np.add.at(sep_sum, inv, contact_prob)
    np.add.at(sep_cnt, inv, 1)
    expected_at_sep = (sep_sum / sep_cnt)[inv]           # 各ペアの「同じ間隔なら平均どれくらい接触するか」
    local_enrichment = contact_prob / np.maximum(expected_at_sep, 1e-12)

    cats = fp["categories"]
    cat_list = sorted(set(cats))
    global_mean = contact_prob.mean()
    rows = []
    for a in cat_list:
        for b in cat_list:
            if a > b:
                continue
            mask = ((cats[ri] == a) & (cats[rj] == b)) | \
                   ((cats[ri] == b) & (cats[rj] == a))
            if mask.sum() == 0:
                continue
            obs = contact_prob[mask].mean()
            rows.append(dict(
                cat_i=a, cat_j=b, n_pairs=int(mask.sum()),
                contact_prob=float(obs),
                enrichment_naive=float(obs / global_mean) if global_mean > 0 else np.nan,
                enrichment=float(local_enrichment[mask].mean()),
                mean_seq_separation=float(sep[mask].mean()),
                mean_E_AH=float(mean_e_ah[mask].sum()),
                mean_E_DH=float(mean_e_dh[mask].sum())))
    enrichment = pd.DataFrame(rows).sort_values("enrichment", ascending=False)

    return dict(
        pairs=pairs, contact_prob=contact_prob, n_contacts_frame=n_contacts_frame,
        E_AH_frame=e_ah.sum(axis=1), E_DH_frame=e_dh.sum(axis=1),
        residue_E_AH=res_ah, residue_E_DH=res_dh,
        categories=cats, enrichment=enrichment,
        contact_cutoff_factor=contact_cutoff_factor,
        n_frames_used=t.n_frames)


def plot_interactions(results: dict, inter: dict, outdir: Path):
    """接触マップ・エンリッチメント・エネルギー・形状記述子を 1 枚にまとめる。"""
    systems = list(inter)
    n_res = int(inter[systems[0]]["pairs"].max()) + 1
    fig = plt.figure(figsize=(14, 8.6), dpi=150)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.34)

    # --- (0,0) 接触確率マップ差分 (系が2つの場合) / 1系なら単体表示 ---
    ax = fig.add_subplot(gs[0, 0])
    if len(systems) >= 2:
        a, b = systems[0], systems[1]
        Ma = np.zeros((n_res, n_res)); Mb = np.zeros((n_res, n_res))
        Ma[inter[a]["pairs"][:, 0], inter[a]["pairs"][:, 1]] = inter[a]["contact_prob"]
        Mb[inter[b]["pairs"][:, 0], inter[b]["pairs"][:, 1]] = inter[b]["contact_prob"]
        Ma += Ma.T; Mb += Mb.T
        diff = Mb - Ma
        v = np.nanpercentile(np.abs(diff), 99) or 1.0
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Δ contact prob.")
        for p in results[b].get("phospho_sites", []):
            ax.axhline(p - 1, color="k", lw=0.2, alpha=0.3)
            ax.axvline(p - 1, color="k", lw=0.2, alpha=0.3)
        ax.set_title(f"接触確率差分: {results[b]['label']} − {results[a]['label']}",
                    fontsize=9)
    else:
        s = systems[0]
        M = np.zeros((n_res, n_res))
        M[inter[s]["pairs"][:, 0], inter[s]["pairs"][:, 1]] = inter[s]["contact_prob"]
        M += M.T
        im = ax.imshow(M, cmap="viridis", vmin=0, origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, label="contact prob.")
        ax.set_title(f"接触確率マップ: {results[s]['label']}", fontsize=9)
    ax.set_xlabel("residue i"); ax.set_ylabel("residue j")

    # --- (0,1) 電荷タイプ別エンリッチメント ---
    ax = fig.add_subplot(gs[0, 1])
    # ラベルは全系の和集合を使う (例: unphos には 'phospho' カテゴリの
    # 残基が存在しないため、unphos だけを基準にすると phospho 絡みの
    # 系列が抜け落ちてしまう)
    enr_tables = {}
    all_pair_labels = set()
    for s in systems:
        e = inter[s]["enrichment"].copy()
        e["pair_label"] = e["cat_i"] + "-" + e["cat_j"]
        e = e.set_index("pair_label")
        enr_tables[s] = e
        all_pair_labels |= set(e.index)
    # phos 側 (存在すれば最後の系) の enrichment 順に並べ、無ければ平均順
    sort_key_system = systems[-1] if systems[-1] in enr_tables else systems[0]
    order_vals = enr_tables[sort_key_system]["enrichment"].reindex(all_pair_labels)
    all_labels = list(order_vals.sort_values(ascending=False,
                                             na_position="last").index)
    width = 0.8 / max(len(systems), 1)
    for k, s in enumerate(systems):
        vals = enr_tables[s]["enrichment"].reindex(all_labels)
        x = np.arange(len(all_labels)) + k * width
        ax.bar(x, vals, width=width, color=PALETTE.get(s), label=results[s]["label"])
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(all_labels)) + width * (len(systems) - 1) / 2)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("接触エンリッチメント (配列間隔補正済み)")
    ax.set_title("電荷タイプ別の接触しやすさ", fontsize=9)
    ax.legend(fontsize=7)

    # --- (0,2) 残基ごとのエネルギー寄与 (差分) ---
    ax = fig.add_subplot(gs[0, 2])
    if len(systems) >= 2:
        a, b = systems[0], systems[1]
        d_ah = inter[b]["residue_E_AH"] - inter[a]["residue_E_AH"]
        d_dh = inter[b]["residue_E_DH"] - inter[a]["residue_E_DH"]
        x = np.arange(1, n_res + 1)
        ax.plot(x, d_dh, color="crimson", lw=0.9, label="ΔE 静電 (DH)")
        ax.plot(x, d_ah, color="steelblue", lw=0.9, label="ΔE 疎水性 (AH)")
        for p in results[b].get("phospho_sites", []):
            ax.axvline(p, color="0.6", lw=0.4, alpha=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"残基ごとのエネルギー変化 ({results[b]['label']}−{results[a]['label']})",
                    fontsize=8.5)
    else:
        s = systems[0]
        x = np.arange(1, n_res + 1)
        ax.plot(x, inter[s]["residue_E_DH"], color="crimson", lw=0.9, label="静電 (DH)")
        ax.plot(x, inter[s]["residue_E_AH"], color="steelblue", lw=0.9, label="疎水性 (AH)")
        ax.set_title(f"残基ごとのエネルギー寄与: {results[s]['label']}", fontsize=9)
    ax.set_xlabel("residue"); ax.set_ylabel("エネルギー (kJ/mol)")
    ax.legend(fontsize=7)

    # --- (1,0) 相互作用エネルギーの分布 ---
    ax = fig.add_subplot(gs[1, 0])
    for s in systems:
        c = PALETTE.get(s)
        ax.hist(inter[s]["E_DH_frame"], bins=40, density=True, alpha=0.35,
               color=c, label=f"{results[s]['label']} (静電)")
    ax.set_xlabel("$E_{DH}$ (kJ/mol, 静電)"); ax.set_ylabel("density")
    ax.set_title("静電相互作用エネルギーの分布", fontsize=9)
    ax.legend(fontsize=7)

    # --- (1,1) 接触数 vs Rg ---
    ax = fig.add_subplot(gs[1, 1])
    for s in systems:
        c = PALETTE.get(s)
        nC = inter[s]["n_contacts_frame"]
        rg_strided = inter[s].get("rg_strided")
        if rg_strided is not None and len(rg_strided) == len(nC):
            ax.scatter(rg_strided, nC, s=4, alpha=0.25, color=c,
                      label=results[s]["label"])
    ax.set_xlabel("$R_g$ (nm)"); ax.set_ylabel("分子内接触数")
    ax.set_title("接触数と Rg の関係", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)

    # --- (1,2) 形状記述子 (asphericity vs kappa^2) ---
    ax = fig.add_subplot(gs[1, 2])
    for s in systems:
        c = PALETTE.get(s)
        sh = results[s]["shape"]
        ax.scatter(sh["kappa2"][::5], sh["asphericity"][::5] / sh["rg2"][::5],
                  s=3, alpha=0.15, color=c, rasterized=True)
        ax.scatter([], [], color=c, label=results[s]["label"])
    ax.set_xlabel(r"形状異方性 $\kappa^2$ (0=球, 1=棒状)")
    ax.set_ylabel(r"非対称度 $b/R_g^2$")
    ax.set_title("鎖の形状", fontsize=9)
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle("分子内相互作用の解析", fontsize=13, y=1.01)
    fig.savefig(outdir / "interactions.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# 極端構造の抽出
# ==========================================================================
def save_extremes(traj, rg, outdir: Path, system: str, n_each: int = 5):
    """最もコンパクト / 最も伸びた構造を PDB で保存する。"""
    outdir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(rg)
    compact_idx = order[:n_each]
    extended_idx = order[::-1][:n_each]

    # 単一構造 (最頻値付近 = 代表構造) も出す
    kde = gaussian_kde(rg)
    grid = np.linspace(rg.min(), rg.max(), 400)
    mode_rg = grid[np.argmax(kde(grid))]
    typical_idx = int(np.argmin(np.abs(rg - mode_rg)))

    ref = traj[int(compact_idx[0])]
    records = []
    for tag, idxs in [("compact", compact_idx), ("extended", extended_idx),
                      ("typical", [typical_idx])]:
        for rank, i in enumerate(idxs, start=1):
            fr = traj[int(i)]
            fr.superpose(ref)          # 見やすいように重ね合わせ
            name = f"{system}_{tag}{rank:02d}_Rg{rg[int(i)]:.2f}nm.pdb"
            fr.save_pdb(str(outdir / name))
            records.append(dict(system=system, kind=tag, rank=rank,
                                frame=int(i), Rg_nm=float(rg[int(i)]),
                                file=name))
    # まとめて 1 本のマルチモデル PDB にも
    sel = list(compact_idx) + [typical_idx] + list(extended_idx)
    multi = traj[[int(i) for i in sel]]
    multi.superpose(ref)
    multi.save_pdb(str(outdir / f"{system}_extremes_multimodel.pdb"))
    return pd.DataFrame(records), mode_rg


# ==========================================================================
# プロット
# ==========================================================================
def plot_all(results: dict, cfg, outdir: Path):
    systems = list(results)
    fig = plt.figure(figsize=(13.5, 9.5), dpi=150)
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)

    # --- Rg 分布 ---
    ax = fig.add_subplot(gs[0, :2])
    for s in systems:
        R = results[s]
        c = PALETTE.get(s)
        ax.hist(R["rg"], bins=cfg.get("bins", 60), density=True, alpha=0.25, color=c)
        grid = np.linspace(R["rg"].min(), R["rg"].max(), 500)
        ax.plot(grid, gaussian_kde(R["rg"])(grid), color=c, lw=2,
                label=f"{R['label']}: $\\langle R_g \\rangle$ = "
                      f"{R['rg_mean']:.2f} $\\pm$ {R['rg_err']:.2f} nm")
        ax.axvline(R["rg_mean"], color=c, ls="--", lw=1)
        ax.axvline(R["rg_min"], color=c, ls=":", lw=1)
        ax.axvline(R["rg_max"], color=c, ls=":", lw=1)
    ax.set_xlabel("$R_g$ (nm)"); ax.set_ylabel("$p(R_g)$")
    ax.set_title(LBL["dist_title"]); ax.legend(fontsize=8)

    # --- CDF ---
    ax = fig.add_subplot(gs[0, 2])
    for s in systems:
        R = results[s]
        x = np.sort(R["rg"])
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=PALETTE.get(s),
                lw=1.6, label=R["label"])
    ax.set_xlabel("$R_g$ (nm)"); ax.set_ylabel(LBL["cdf_y"])
    ax.set_title(LBL["cdf_title"]); ax.legend(fontsize=7)

    # --- Rg 時系列 ---
    ax = fig.add_subplot(gs[1, :2])
    for s in systems:
        R = results[s]
        off = 0.0
        for k, (rg_r, dtn) in enumerate(zip(R["rg_per_rep"], R["dt_per_rep"])):
            t = off + np.arange(len(rg_r)) * dtn
            ax.plot(t, rg_r, lw=0.4, alpha=0.75, color=PALETTE.get(s),
                    label=R["label"] if k == 0 else None)
            off = t[-1] + dtn
            ax.axvline(off, color="0.85", lw=0.5)
        ax.axhline(R["rg_mean"], color=PALETTE.get(s), ls="--", lw=1)
    ax.set_xlabel(LBL["ts_x"]); ax.set_ylabel("$R_g$ (nm)")
    ax.set_title(LBL["ts_title"]); ax.legend(fontsize=8)

    # --- Ree 分布 ---
    ax = fig.add_subplot(gs[1, 2])
    for s in systems:
        R = results[s]
        grid = np.linspace(R["ree"].min(), R["ree"].max(), 400)
        ax.plot(grid, gaussian_kde(R["ree"])(grid), color=PALETTE.get(s), lw=1.8,
                label=f"{R['label']}: {R['ree_mean']:.1f} nm")
    ax.set_xlabel("$R_{ee}$ (nm)"); ax.set_ylabel("$p(R_{ee})$")
    ax.set_title(LBL["ree_title"]); ax.legend(fontsize=7)

    # --- 内部スケーリング ---
    ax = fig.add_subplot(gs[2, 0])
    for s in systems:
        R = results[s]
        ax.plot(R["ij"], R["dij"], color=PALETTE.get(s), lw=1.6,
                label=f"{R['label']}: $\\nu$ = {R['nu']:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|i - j|")
    ax.set_ylabel(r"$\sqrt{\langle d_{ij}^2 \rangle}$ (nm)")
    ax.set_title(LBL["scal_title"]); ax.legend(fontsize=7)

    # --- 距離マップ差分 ---
    ax = fig.add_subplot(gs[2, 1])
    if len(systems) >= 2:
        a, b = systems[0], systems[1]
        diff = results[b]["dmap"] - results[a]["dmap"]
        v = float(np.nanpercentile(np.abs(diff), 99)) or 1.0
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label=r"$\Delta \langle d_{ij} \rangle$ (nm)")
        for pnum in results[b].get("phospho_sites", []):
            ax.axhline(pnum - 1, color="k", lw=0.25, alpha=0.3)
            ax.axvline(pnum - 1, color="k", lw=0.25, alpha=0.3)
        ax.set_title(f"{results[b]['label']} - {results[a]['label']}", fontsize=9)
        ax.set_xlabel(LBL["dmap_x"]); ax.set_ylabel(LBL["dmap_y"])

    # --- テキストまとめ ---
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    lines = [LBL["summary"], ""]
    for s in systems:
        R = results[s]
        lines += [f"* {R['label']}  (n={len(R['rg'])} {LBL['nframes']})",
                  f"   <Rg>  = {R['rg_mean']:.3f} +/- {R['rg_err']:.3f} nm",
                  f"   SD    = {R['rg_sd']:.3f} nm  (n_eff~{R['n_eff']:.0f})",
                  f"   min/max = {R['rg_min']:.2f} / {R['rg_max']:.2f} nm",
                  f"   <Ree> = {R['ree_mean']:.2f} nm,  nu = {R['nu']:.3f}",
                  ""]
    if len(systems) >= 2:
        a, b = systems[0], systems[1]
        dr = results[b]["rg_mean"] - results[a]["rg_mean"]
        lines += [f"* {LBL['change']}",
                  f"   dRg = {dr:+.3f} nm ({100 * dr / results[a]['rg_mean']:+.1f} %)",
                  f"   dRg/Rg(ph) = {dr / results[b]['rg_mean']:+.4f}",
                  f"   {LBL['pos_note']}"]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7.6,
            transform=ax.transAxes,
            fontfamily=plt.rcParams["font.family"][0]
            if isinstance(plt.rcParams["font.family"], list)
            else plt.rcParams["font.family"])

    fig.suptitle(LBL["suptitle"], fontsize=13, y=0.985)
    out = outdir / "summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rg_only(results, cfg, outdir: Path):
    """論文図に近い、Rg 分布だけのきれいな 1 枚。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=200)
    for s, R in results.items():
        c = PALETTE.get(s)
        grid = np.linspace(R["rg"].min() * 0.93, R["rg"].max() * 1.07, 600)
        p = gaussian_kde(R["rg"])(grid)
        ax.fill_between(grid, p, alpha=0.20, color=c)
        ax.plot(grid, p, color=c, lw=2.2,
                label=f"{R['label']}   $\\langle R_g \\rangle$ = "
                      f"{R['rg_mean']:.2f} $\\pm$ {R['rg_err']:.2f} nm")
        ax.axvline(R["rg_mean"], color=c, ls="--", lw=1.2)
        ax.plot([R["rg_min"], R["rg_max"]], [0, 0], marker="v", ls="none",
                color=c, ms=7, clip_on=False)
    ax.set_xlabel("$R_g$ (nm)", fontsize=12)
    ax.set_ylabel(LBL["prob"], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(LBL["single_title"], fontsize=12)
    fig.savefig(outdir / "rg_distribution.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# メイン
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--systems", nargs="*", default=None)
    p.add_argument("--outdir", default="analysis")
    p.add_argument("--equil-ns", type=float, default=None,
                   help="捨てる冒頭の時間 (既定は config の equil_time_ns)")
    p.add_argument("--stride-map", type=int, default=5,
                   help="距離マップ計算のフレーム間引き")
    args = p.parse_args()

    cfg_all = yaml.safe_load(open(args.config))
    base = Path(args.config).parent
    root = base / cfg_all.get("outdir", "runs")
    acfg = cfg_all.get("analysis", {})
    systems = args.systems or acfg.get("compare") or list(cfg_all["systems"])
    equil_ns = args.equil_ns if args.equil_ns is not None \
        else cfg_all["simulation"].get("equil_time_ns", 0.0)

    outdir = base / args.outdir
    (outdir / "structures").mkdir(parents=True, exist_ok=True)

    setup_fonts()

    results, all_extremes, trajs = {}, [], {}
    for s in systems:
        print(f"\n=== {s} ===")
        traj, per_traj, per_rep, meta = load_system(root, s, equil_ns)
        trajs[s] = traj      # PCA で系をまたいで使うため保持しておく
        masses = np.array(meta["masses"], dtype=float)

        rg = compute_rg(traj, masses)
        ree = compute_ree(traj)
        rg_mean, rg_err, n_eff, _, _ = block_error(rg)
        ij, dij, R0, nu, nu_err = internal_scaling(traj, stride=args.stride_map)
        dmap = mean_distance_map(traj, stride=args.stride_map)
        shape = gyration_tensor_shape(traj, masses)

        ext_df, mode_rg = save_extremes(
            traj, rg, outdir / "structures", s,
            n_each=int(acfg.get("n_extreme_structures", 5)))
        all_extremes.append(ext_df)

        # レプリカ別
        rg_per_rep = [compute_rg(t, masses) for t in per_traj]
        results[s] = dict(
            label=cfg_all["systems"][s].get("label", s),
            rg=rg, ree=ree, rg_mean=rg_mean, rg_err=rg_err,
            rg_sd=float(rg.std(ddof=1)), n_eff=n_eff,
            rg_min=float(rg.min()), rg_max=float(rg.max()), rg_mode=float(mode_rg),
            ree_mean=float(ree.mean()),
            ij=ij, dij=dij, nu=float(nu), nu_err=nu_err, R0=float(R0),
            dmap=dmap, phospho_sites=meta.get("phospho_sites", []),
            shape=shape,
            rg_per_rep=rg_per_rep,
            dt_per_rep=[r["dt_frame_ns"] for r in per_rep],
            per_rep=per_rep, meta=meta)

        # 時系列 CSV
        off, rows = 0.0, []
        for r_i, (rr, pr) in enumerate(zip(rg_per_rep, per_rep)):
            t = np.arange(len(rr)) * pr["dt_frame_ns"] + equil_ns
            rows.append(pd.DataFrame(dict(replica=pr["replica"], time_ns=t,
                                          Rg_nm=rr)))
        ts = pd.concat(rows, ignore_index=True)
        ts["Ree_nm"] = ree
        ts.to_csv(outdir / f"timeseries_{s}.csv", index=False)

        print(f"  <Rg> = {rg_mean:.3f} ± {rg_err:.3f} nm "
              f"(SD {rg.std(ddof=1):.3f}, n_eff≈{n_eff:.0f})")
        print(f"  Rg range = [{rg.min():.2f}, {rg.max():.2f}] nm, mode {mode_rg:.2f}")
        print(f"  <Ree> = {ree.mean():.2f} nm,  nu = {nu:.3f} ± {nu_err:.3f}")

    # --- 図 ---
    png = plot_all(results, acfg, outdir)
    plot_rg_only(results, acfg, outdir)
    pd.concat(all_extremes, ignore_index=True).to_csv(
        outdir / "extreme_structures.csv", index=False)

    # --- 主成分分析 (系を 2 つ以上比較するときだけ意味があるが、1 系でも可) ---
    pca_res = None
    if acfg.get("pca", True):
        try:
            pca_res = run_pca(results, trajs, acfg, outdir)
            plot_pca(pca_res, results, outdir)
            print(f"\nPC1 と Rg の相関: "
                  + ", ".join(f"{s}={c:+.2f}" for s, c in pca_res["corr"].items()))
        except ImportError:
            print("\n(scikit-learn が無いため PCA をスキップしました。"
                  "'pip install scikit-learn' で有効になります)")

    # --- 分子内相互作用の解析 ---
    inter = None
    if acfg.get("interactions", True):
        print("\n=== 分子内相互作用 ===")
        inter = {}
        cutoff_factor = float(acfg.get("contact_cutoff_factor", 1.5))
        for s in systems:
            r = interaction_analysis(trajs[s], results[s]["meta"],
                                     stride=args.stride_map,
                                     contact_cutoff_factor=cutoff_factor)
            r["rg_strided"] = results[s]["rg"][::args.stride_map][:len(r["n_contacts_frame"])]
            inter[s] = r

            e_ah_mean, e_ah_err, *_ = block_error(r["E_AH_frame"])
            e_dh_mean, e_dh_err, *_ = block_error(r["E_DH_frame"])
            nC_mean, nC_err, *_ = block_error(r["n_contacts_frame"].astype(float))
            print(f"  [{s}] E_AH(疎水性) = {e_ah_mean:9.1f} ± {e_ah_err:5.1f} kJ/mol   "
                  f"E_DH(静電) = {e_dh_mean:8.1f} ± {e_dh_err:5.1f} kJ/mol   "
                  f"接触数 = {nC_mean:6.1f} ± {nC_err:4.1f}")
            print(f"        SCD (配列電荷デコレーション) = "
                  f"{results[s]['meta']['composition'].get('SCD', float('nan')):+.3f}")

            top = r["enrichment"].head(3)
            bot = r["enrichment"].tail(3)
            print(f"        接触エンリッチメント上位: "
                  + ", ".join(f"{row.cat_i}-{row.cat_j}={row.enrichment:.2f}"
                              for row in top.itertuples()))
            print(f"        接触エンリッチメント下位: "
                  + ", ".join(f"{row.cat_i}-{row.cat_j}={row.enrichment:.2f}"
                              for row in bot.itertuples()))

            r["enrichment"].to_csv(outdir / f"contact_enrichment_{s}.csv", index=False)
            np.save(outdir / f"contact_probability_{s}.npy", r["contact_prob"])
            pd.DataFrame({
                "residue": np.arange(1, len(r["residue_E_AH"]) + 1),
                "token": results[s]["meta"]["sequence_tokens"],
                "E_AH_kJmol": r["residue_E_AH"],
                "E_DH_kJmol": r["residue_E_DH"],
            }).to_csv(outdir / f"residue_energy_{s}.csv", index=False)

        plot_interactions(results, inter, outdir)

    # --- 要約テーブル ---
    summ = pd.DataFrame([{
        "system": s, "label": R["label"], "n_frames": len(R["rg"]),
        "Rg_mean_nm": R["rg_mean"], "Rg_err_nm": R["rg_err"],
        "Rg_sd_nm": R["rg_sd"], "n_eff": R["n_eff"],
        "Rg_min_nm": R["rg_min"], "Rg_max_nm": R["rg_max"],
        "Rg_mode_nm": R["rg_mode"], "Ree_mean_nm": R["ree_mean"],
        "nu": R["nu"], "nu_err": R["nu_err"],
        "n_phospho": len(R["phospho_sites"]),
        "asphericity_mean": float(np.mean(R["shape"]["asphericity"])),
        "kappa2_mean": float(np.mean(R["shape"]["kappa2"])),
        "SCD": R["meta"]["composition"].get("SCD", np.nan),
        "E_AH_mean_kJmol": (block_error(inter[s]["E_AH_frame"])[0]
                            if inter else np.nan),
        "E_DH_mean_kJmol": (block_error(inter[s]["E_DH_frame"])[0]
                            if inter else np.nan),
        "n_contacts_mean": (block_error(inter[s]["n_contacts_frame"].astype(float))[0]
                            if inter else np.nan),
    } for s, R in results.items()])
    if len(summ) >= 2:
        a, b = summ.iloc[0], summ.iloc[1]
        summ.attrs["dRg"] = b.Rg_mean_nm - a.Rg_mean_nm
    summ.to_csv(outdir / "summary.csv", index=False)

    print("\n" + "=" * 70)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(summ) >= 2:
        dr = summ.iloc[1].Rg_mean_nm - summ.iloc[0].Rg_mean_nm
        print(f"\nΔRg = {dr:+.3f} nm  "
              f"({100 * dr / summ.iloc[0].Rg_mean_nm:+.1f} %),  "
              f"ΔRg/Rg(ph) = {dr / summ.iloc[1].Rg_mean_nm:+.4f}")
    print("=" * 70)
    print(f"\n図       : {png}")
    print(f"           {outdir / 'rg_distribution.png'}")
    if inter is not None:
        print(f"           {outdir / 'interactions.png'}")
    if pca_res is not None:
        print(f"           {outdir / 'pca.png'}")
    print(f"構造     : {outdir / 'structures'}/  (*_compact*, *_extended*, *_typical*)")
    print(f"テーブル : {outdir / 'summary.csv'}, {outdir / 'extreme_structures.csv'}")


if __name__ == "__main__":
    main()
