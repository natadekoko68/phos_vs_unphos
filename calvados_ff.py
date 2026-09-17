"""
calvados_ff.py
==============
CALVADOS 2 の残基パラメータに、リン酸化セリン / スレオニン (pSer, pThr) を追加する。

パラメータ化は Rauh et al., Biophys. J. 125, 396-405 (2026)
"A coarse-grained model for simulations of phosphorylated disordered proteins"
に従う:

  1. bead サイズ : 非リン酸化残基の体積を +0.041 nm^3 する
                   V = (pi/6) * sigma^3  ->  sigma_p = (6*(V + 0.041)/pi)^(1/3)
  2. 電荷        : pH 依存 (論文 Eq. 7)
                   q = -1 - 1/(1 + 10^(pKa - pH)),  pKa(pSer)=6.01, pKa(pThr)=6.30
  3. stickiness  : lambda_pX = lambda_X + dlambda,  dlambda = -0.37
                   -> lambda_pSer = 0.09, lambda_pThr ~ 0  (論文の最終値)

配列は 1 文字コードのトークン列として扱う。リン酸化残基は "pS" / "pT" と書く:
    "...LCLpSPApSSG..."  ->  ['L','C','L','pS','P','A','pS','S','G', ...]
"""

from __future__ import annotations

import re
from io import StringIO

import numpy as np
import pandas as pd

# --- 論文で決められた定数 -------------------------------------------------
DLAMBDA_PHOSPHO = -0.37       # stickiness のシフト (Rauh et al. の最終値)
DVOLUME_PHOSPHO = 0.041       # nm^3, bead 体積の増分
DMW_PHOSPHO = 79.98           # Da, HPO3 の付加

PKA_HIS = 6.00
PKA_PHOSPHO = {"pS": 6.01, "pT": 6.30, "pY": 5.96}
THREE_PHOSPHO = {"pS": "SEP", "pT": "TPO", "pY": "PTR"}

# --- CALVADOS 2 の残基パラメータ (KULL-Centre/CALVADOS の residues.csv と同一) ---
_CALVADOS2 = """\
one,three,MW,lambdas,sigmas,q
R,ARG,156.19,0.7307624767517166,0.656,1
D,ASP,115.09,0.0416040480605567,0.558,-1
N,ASN,114.10,0.4255859009787713,0.568,0
E,GLU,129.11,0.0006935460962935,0.592,-1
K,LYS,128.17,0.1790211738990582,0.636,1
H,HIS,137.14,0.4663667290557992,0.608,0
Q,GLN,128.13,0.3934318551056041,0.602,0
S,SER,87.08,0.4625416811611541,0.518,0
C,CYS,103.14,0.5615435099141777,0.548,0
G,GLY,57.05,0.7058843733666401,0.450,0
T,THR,101.11,0.3713162976273964,0.562,0
A,ALA,71.07,0.2743297969040348,0.504,0
M,MET,131.20,0.5308481134337497,0.618,0
Y,TYR,163.18,0.9774611449343455,0.646,0
V,VAL,99.13,0.2083769608174481,0.586,0
W,TRP,186.22,0.9893764740371644,0.678,0
L,LEU,113.16,0.6440005007782226,0.618,0
I,ILE,113.16,0.5423623610671892,0.618,0
P,PRO,97.12,0.3593126576364644,0.556,0
F,PHE,147.18,0.8672358982062975,0.636,0
"""


def _grow_sigma(sigma: float, dV: float = DVOLUME_PHOSPHO) -> float:
    """bead 体積を dV [nm^3] 増やしたときの新しい直径 [nm]。"""
    volume = np.pi / 6.0 * float(sigma) ** 3 + dV
    return float((6.0 * volume / np.pi) ** (1.0 / 3.0))


def load_residues(dlambda: float = DLAMBDA_PHOSPHO,
                  grow_beads: bool = True) -> pd.DataFrame:
    """CALVADOS 2 + pSer/pThr/pTyr の残基テーブルを返す。

    列: three, MW, lambdas, sigmas, q, pKa, ionization
    index は 1 文字トークン ('S', 'pS', ...)。
    'q' は形式電荷。pH 依存の残基は ionization ('his' / 'phospho') を持ち、
    apply_ph() で実効電荷に変換される。

    Parameters
    ----------
    dlambda : リン酸化による stickiness シフト。論文の最終値は -0.37。
    grow_beads : False にすると bead サイズを変えない (charge model 用)。
    """
    df = pd.read_csv(StringIO(_CALVADOS2)).set_index("one")
    df["pKa"] = np.nan
    df["ionization"] = ""
    df.loc["H", "pKa"] = PKA_HIS
    df.loc["H", "ionization"] = "his"

    for ptok, parent in [("pS", "S"), ("pT", "T"), ("pY", "Y")]:
        base = df.loc[parent]
        df.loc[ptok] = {
            "three": THREE_PHOSPHO[ptok],
            "MW": float(base.MW) + DMW_PHOSPHO,
            # stickiness を負にすると Ashbaugh-Hatch が非物理的になるため 0 で下限を切る
            "lambdas": max(0.0, float(base.lambdas) + dlambda),
            "sigmas": _grow_sigma(base.sigmas) if grow_beads else float(base.sigmas),
            "q": 0.0,                       # 実効電荷は pKa から計算
            "pKa": PKA_PHOSPHO[ptok],
            "ionization": "phospho",
        }
    return df


def apply_ph(residues: pd.DataFrame, pH: float) -> pd.DataFrame:
    """Henderson-Hasselbalch で pH 依存の部分電荷を入れたテーブルを返す。

    His    : q = +1 / (1 + 10^(pH - pKa))
    phospho: q = -1 - 1/(1 + 10^(pKa - pH))   (論文 Eq. 7, リン酸基 2 段階目の解離)
    """
    r = residues.copy()
    for tok in r.index:
        kind = r.loc[tok, "ionization"]
        if kind == "his":
            r.loc[tok, "q"] = 1.0 / (1.0 + 10 ** (pH - r.loc[tok, "pKa"]))
        elif kind == "phospho":
            r.loc[tok, "q"] = -1.0 - 1.0 / (1.0 + 10 ** (r.loc[tok, "pKa"] - pH))
    return r


# --------------------------------------------------------------------------
# 配列のトークン化
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"p[STY]|[ACDEFGHIKLMNPQRSTVWY]")


def tokenize(sequence: str) -> list[str]:
    """'ACpSDE' -> ['A','C','pS','D','E'] 。空白・改行は無視する。"""
    seq = "".join(sequence.split())
    # 'ps' / 'PS' のような入力の揺れを吸収する
    seq = re.sub(r"p([sty])", lambda m: "p" + m.group(1).upper(), seq)
    seq = re.sub(r"P(?=[STY](?![A-Za-z]*$))", "P", seq)  # no-op: 明示のため

    tokens, i = [], 0
    while i < len(seq):
        m = _TOKEN_RE.match(seq, i)
        if m is None:
            raise ValueError(
                f"配列の {i} 文字目 '{seq[i]}' を解釈できません "
                f"(周辺: ...{seq[max(0, i - 10):i + 10]}...)")
        tokens.append(m.group(0))
        i = m.end()
    return tokens


def unphosphorylated(tokens: list[str]) -> list[str]:
    """pS -> S のように脱リン酸化したトークン列を返す。"""
    return [t[-1] if t.startswith("p") else t for t in tokens]


def phospho_sites(tokens: list[str]) -> list[int]:
    """リン酸化サイトの 1-based 残基番号。"""
    return [i + 1 for i, t in enumerate(tokens) if t.startswith("p")]


def sequence_string(tokens: list[str]) -> str:
    return "".join(tokens)


# --------------------------------------------------------------------------
# 論文 Fig.3 の比較モデル
# --------------------------------------------------------------------------
def build_model(tokens: list[str], mode: str = "phospho",
                dlambda: float = DLAMBDA_PHOSPHO) -> tuple[list[str], pd.DataFrame]:
    """モードに応じたトークン列と残基テーブルを返す。

    mode:
      'phospho' : 論文の data-driven モデル (dlambda=-0.37, bead 拡大, q ~ -2)
      'unphos'  : すべて脱リン酸化 (baseline)
      'charge'  : サイズ・stickiness は Ser/Thr のまま電荷だけ ~-2 (論文 "charge model")
      'mimetic' : pSer -> Asp, pThr -> Glu (phosphomimetic)
    """
    mode = mode.lower()
    if mode in ("phospho", "phos"):
        return list(tokens), load_residues(dlambda=dlambda)
    if mode == "unphos":
        return unphosphorylated(tokens), load_residues(dlambda=dlambda)
    if mode == "mimetic":
        mapping = {"pS": "D", "pT": "E", "pY": "E"}
        return [mapping.get(t, t) for t in tokens], load_residues(dlambda=dlambda)
    if mode == "charge":
        return list(tokens), load_residues(dlambda=0.0, grow_beads=False)
    raise ValueError(f"unknown mode: {mode!r}")


def summarize(tokens: list[str], residues: pd.DataFrame, pH: float) -> dict:
    """配列の電荷組成などの要約を返す (sanity check 用)。"""
    r = apply_ph(residues, pH)
    q = np.array([r.loc[t, "q"] for t in tokens], dtype=float)
    q[0] += 1.0     # N 末端アミノ基
    q[-1] -= 1.0    # C 末端カルボキシル基
    n = len(tokens)
    return {
        "N_residues": n,
        "net_charge": float(q.sum()),
        "NCPR": float(q.sum() / n),
        "FCR": float(np.abs(q).sum() / n),
        "n_phospho": len(phospho_sites(tokens)),
        "MW_kDa": float(sum(r.loc[t, "MW"] for t in tokens) / 1000.0 + 0.018),
        "mean_lambda": float(np.mean([r.loc[t, "lambdas"] for t in tokens])),
        "SCD": scd(tokens, q),
    }


# ==========================================================================
# 静電・非イオン性相互作用の物理量 (分子内相互作用の解析用)
#
# simulate.py がシミュレーション本体で使っている式と完全に同じものを、
# OpenMM を必要としない純粋な numpy 関数として再実装する。
# analyze.py はこれを使って、実際に走らせたトラジェクトリから
# 相互作用エネルギー (AH: 疎水性/排除体積、DH: 塩遮蔽静電) を
# フレームごと・残基対ごとに計算する。
# ==========================================================================
KB_KJ = 8.3145e-3          # kJ/mol/K


def dielectric_water(T: float) -> float:
    """水の誘電率の経験式 (Akerlof & Oshry 1950; simulate.py と同一の式)。"""
    return (5321.0 / T + 233.76 - 0.9297 * T
            + 0.1417e-2 * T * T - 0.8292e-6 * T ** 3)


def debye_and_charges(tokens: list[str], residues: pd.DataFrame,
                      temperature: float, ionic_strength: float, pH: float):
    """Debye-Hueckel 項に必要な量一式を返す。

    simulate.py の yukawa_params() と完全に同じ計算。
    OpenMM 側のエネルギー式  q1*q2*(exp(-kappa*r)/r - shift)  に合わせて、
    各粒子に q_i = charge_i * sqrt(lB * kT) を持たせる形にしてある。

    Returns
    -------
    yu_eps : (N,) 各残基の "実効電荷" (上記の q_i)
    kappa  : 逆デバイ長 [1/nm]
    charges: (N,) 末端補正込みの形式電荷
    lB     : Bjerrum 長 [nm]
    """
    r = apply_ph(residues, pH)
    charges = np.array([float(r.loc[t, "q"]) for t in tokens])
    charges[0] += 1.0       # N 末端 (NH3+)
    charges[-1] -= 1.0      # C 末端 (COO-)

    kT = KB_KJ * temperature
    eps_w = dielectric_water(temperature)
    lB = 1.6021766 ** 2 / (4 * np.pi * 8.854188 * eps_w) * 6.022 * 1000 / kT
    yu_eps = charges * np.sqrt(lB * kT)
    kappa = np.sqrt(8 * np.pi * lB * ionic_strength * 6.022 / 10)
    return yu_eps, float(kappa), charges, float(lB)


def ashbaugh_hatch_energy(r: np.ndarray, sigma_ij: np.ndarray,
                          lambda_ij: np.ndarray, eps: float,
                          rc: float) -> np.ndarray:
    """Ashbaugh-Hatch ポテンシャル (truncated & shifted) をベクトル計算する。

    simulate.py の CustomNonbondedForce (ah_expr) と完全に同じ式。
    r, sigma_ij, lambda_ij はブロードキャスト可能な形状であればよい。
    r >= rc では 0 を返す。
    """
    r = np.asarray(r, dtype=np.float64)
    s, l = np.asarray(sigma_ij), np.asarray(lambda_ij)
    sr6 = (s / r) ** 6
    lj = 4.0 * eps * (sr6 ** 2 - sr6)
    shift = (s / rc) ** 12 - (s / rc) ** 6

    rmin = 2 ** (1.0 / 6.0) * s
    inner = lj + eps * (1.0 - l)                      # r <= rmin (WCA 部分)
    outer = l * (lj - 4.0 * eps * shift)               # rmin < r < rc
    e = np.where(r <= rmin, inner, outer)
    return np.where(r < rc, e, 0.0)


def yukawa_energy(r: np.ndarray, yi: np.ndarray, yj: np.ndarray,
                  kappa: float, rc: float) -> np.ndarray:
    """塩遮蔽 Debye-Hueckel (Yukawa) ポテンシャルをベクトル計算する。

    simulate.py の CustomNonbondedForce (yu) と完全に同じ式。
    """
    r = np.asarray(r, dtype=np.float64)
    shift = np.exp(-kappa * rc) / rc
    e = yi * yj * (np.exp(-kappa * r) / r - shift)
    return np.where(r < rc, e, 0.0)


# --------------------------------------------------------------------------
# 電荷タイプによる残基分類 (分子内相互作用の由来を調べるため)
# --------------------------------------------------------------------------
def classify_residues(tokens: list[str]) -> np.ndarray:
    """各残基を側鎖の種類で分類する ('phospho' / 'acidic' / 'basic' / 'his' / 'other')。

    His は pH によって電荷が変わるため独立カテゴリにしてある。
    """
    cat = []
    for t in tokens:
        if t.startswith("p"):
            cat.append("phospho")
        elif t in ("D", "E"):
            cat.append("acidic")
        elif t in ("K", "R"):
            cat.append("basic")
        elif t == "H":
            cat.append("his")
        else:
            cat.append("other")
    return np.array(cat)


def scd(tokens: list[str], charges: np.ndarray) -> float:
    """配列電荷デコレーション (Sequence Charge Decoration, Sawle & Ghosh 2015)。

        SCD = (1/N) * sum_{i<j} q_i * q_j * sqrt(j - i)

    負に大きいほど正負電荷が配列上でブロック的に分離している (segregated)
    ことを意味し、長距離の分子内静電引力が強く働きやすい配列であることを示す。
    0 に近いほど電荷がよく混ざっている (well-mixed)。
    シミュレーションとは独立に配列だけから計算できる静的な指標。
    """
    q = np.asarray(charges, dtype=float)
    n = len(q)
    i, j = np.triu_indices(n, k=1)
    return float(np.sum(q[i] * q[j] * np.sqrt(j - i)) / n)

