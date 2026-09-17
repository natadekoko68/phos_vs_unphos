# CALVADOS 2 + リン酸化モデルによる IDR の粗視化シミュレーション

Rauh et al., *Biophysical Journal* **125**, 396–405 (2026),
"A coarse-grained model for simulations of phosphorylated disordered proteins"
([doi:10.1016/j.bpj.2025.07.001](https://doi.org/10.1016/j.bpj.2025.07.001))
に基づいて、リン酸化セリン / スレオニンを含む天然変性領域 (IDR) の
1 残基 1 bead 粗視化 MD を行い、慣性半径 $R_g$ の分布を比較するための一式です。

対象は 392 残基の IDR で、

* `unphos` … 非リン酸化体
* `phos`  … 12 箇所の Ser がリン酸化された体 (残基 168, 171, 174, 177, 180, 213, 217, 221, 225, 268, 274, 276)

の 2 系を、GPU 1 枚あたり 1 レプリカ × 1 µs で走らせます。

---

## 1. クイックスタート

```bash
# 環境 (venv + pip。conda は使いません)
./scripts/setup_env.sh
source .venv/bin/activate

# 動作確認 & 速度の見積もり (数十秒)
python scripts/benchmark.py --all-gpus

# ---- HTCondor で流す場合 (NMR box) ----
cd condor
./submit_all.sh                    # config の全 system × n_replicas を投入
condor_q                           # キューの確認
cd ..
python monitor.py --watch          # 進捗ダッシュボード

# ---- condor を使わず 1 台の複数 GPU で流す場合 ----
./scripts/run_local_multigpu.sh
python monitor.py --watch

# ---- 解析 ----
python analyze.py                  # runs/ を読んで analysis/ に出力
```

出力は

| ファイル | 内容 |
|---|---|
| `analysis/rg_distribution.png` | $R_g$ 分布の比較（メイン図） |
| `analysis/summary.png` | 分布・時系列・$R_{ee}$・内部スケーリング・距離マップ差分・数値まとめ |
| `analysis/summary.csv` | `<Rg>`, 誤差, min/max, ν などの表 |
| `analysis/timeseries_*.csv` | $R_g$, $R_{ee}$ の全時系列 |
| `analysis/structures/*.pdb` | **最もコンパクトな構造 / 最も伸びた構造 / 代表構造** |
| `analysis/extreme_structures.csv` | 抽出した構造のフレーム番号と $R_g$ |
| `analysis/pca.png` | **主成分分析**: PC1-PC2 ランドスケープ、寄与率、PC1 の loading map |
| `analysis/pca_projection_*.csv` | 各フレームの PC1..PCn と $R_g$（系ごと） |
| `analysis/structures/*_PC1_low/high*.pdb` | PC1 に沿った極端構造 |
| `analysis/interactions.png` | **分子内相互作用**: 接触確率マップ、電荷タイプ別エンリッチメント、残基ごとのエネルギー、形状記述子 |
| `analysis/contact_probability_*.npy` | 残基対ごとの接触確率（N×N、`np.load` で読む） |
| `analysis/contact_enrichment_*.csv` | 電荷タイプ（酸性/塩基性/リン酸化/His/other）の組ごとの接触エンリッチメントと相互作用エネルギー |
| `analysis/residue_energy_*.csv` | 残基ごとの平均相互作用エネルギー（疎水性 AH・静電 DH） |

---

## 1.5 環境構築の詳細 (pip / venv)

conda は使いません。**NVIDIA ドライバさえ入っていれば CUDA Toolkit の
インストールも不要**です。OpenMM の CUDA プラグイン (`libOpenMMCUDA.so`) は
PyPI のホイールに同梱されており、JIT コンパイルに使う NVRTC もその中にあります。

### 自動セットアップ

```bash
./scripts/setup_env.sh
```

やっていること:

1. Python 3.9 以上かチェック
2. `nvidia-smi` の "CUDA Version:" を読んで `cuda12` / `cuda13` を判定
3. `.venv` を作って `requirements.txt` をインストール
4. `pip install "openmm[cuda12]"` （判定結果に応じて）
5. `Platform` 一覧を出して CUDA が使えるか表示

判定を上書きしたいときは:

```bash
CUDA_EXTRA=cuda13 ./scripts/setup_env.sh    # 明示指定
CUDA_EXTRA=none   ./scripts/setup_env.sh    # CPU のみ
PYTHON=python3.11 ./scripts/setup_env.sh    # 別の Python を使う
./scripts/setup_env.sh /shared/envs/calvados  # venv の場所を変える
```

### 手でやる場合

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install "openmm[cuda12]"        # ← GPU を使うならこれが必須
```

`nvidia-smi` の右上に出る `CUDA Version:` がドライバの対応上限です。
`12.x` なら `cuda12`、`13.x` なら `cuda13`。AMD GPU なら `openmm[hip6]`。

### 確認

```bash
python -m openmm.testInstallation
```

`CUDA` が一覧に出て、Reference との計算値の差が小さければ OK です。

### venv の置き場所について

HTCondor で共有ファイルシステムのクラスタを使う場合、venv は
**全実行ノードから同じパスで見える場所**に置いてください。
`run_sim.sh` は `<project>/.venv` を自動で探しますが、
別の場所なら `VENV_DIR` を export してから `condor_submit` すれば
`getenv = True` 経由でジョブに引き継がれます。

---

## 2. ファイル構成

```
calvados_phospho/
├── config.yaml                  ← 触るのは基本ここだけ
├── requirements.txt
├── calvados_ff.py               CALVADOS 2 + pSer/pThr パラメータ、配列トークナイザ
├── simulate.py                  OpenMM 本体 (1 系 × 1 レプリカ)
├── analyze.py                   Rg 分布・極端構造抽出・作図
├── monitor.py                   全ジョブの進捗ダッシュボード
├── sequences/
│   ├── unphos.fasta
│   └── phos.fasta               リン酸化残基は "pS" / "pT" と書く
├── condor/
│   ├── simulate.sub             HTCondor submit (request_gpus = 1)
│   ├── run_sim.sh               ジョブのラッパー (GPU 割り当ての処理を含む)
│   └── submit_all.sh            jobs.txt を作って condor_submit
└── scripts/
    ├── setup_env.sh             venv + pip で環境構築 (CUDA 版を自動判定)
    ├── benchmark.py             環境チェック + 速度測定
    └── run_local_multigpu.sh    condor 無しで複数 GPU に配る
```

---

## 3. モデルの中身

### 3.1 CALVADOS 2 (Tesei & Lindorff-Larsen 2023)

1 アミノ酸残基 = 1 bead。相互作用は 3 項:

* **結合**: $u_\mathrm{bond}(r) = \tfrac{1}{2}k(r-r_0)^2$、$k = 8033$ kJ/mol/nm², $r_0 = 0.38$ nm
* **非イオン性**: Ashbaugh–Hatch ポテンシャル（truncated & shifted, $r_c = 2$ nm）
  $\epsilon = 0.8368$ kJ/mol、$\lambda$ は 2 残基の stickiness の算術平均、
  $\sigma$ は van der Waals 径の算術平均
* **静電**: 塩遮蔽 Debye–Hückel（$r_c = 4$ nm）、水の誘電率は温度依存の経験式

積分は Langevin（$\Delta t = 10$ fs、摩擦係数 $0.01\ \mathrm{ps}^{-1}$）。
摩擦を弱くしてあるのはサンプリングを速くするためで、CALVADOS の標準設定です。

### 3.2 リン酸化残基 (本論文の寄与)

| | λ | σ (nm) | MW | q (pH 7.4) |
|---|---|---|---|---|
| Ser | 0.4625 | 0.518 | 87.08 | 0 |
| **pSer (SEP)** | **0.0925** | **0.6012** | 167.06 | **−1.961** |
| Thr | 0.3713 | 0.562 | 101.11 | 0 |
| **pThr (TPO)** | **0.0013** | **0.6348** | 181.09 | −1.926 |

* **サイズ**: bead 体積を $+0.041\ \mathrm{nm}^3$ する
  → $\sigma_p = \left(\dfrac{6(V + 0.041)}{\pi}\right)^{1/3}$
* **電荷**: リン酸基の 2 段階目の解離を Henderson–Hasselbalch で扱う（論文 Eq. 7）
  $q = -1 - \dfrac{1}{1 + 10^{\mathrm{p}K_a - \mathrm{pH}}}$、$\mathrm{p}K_a$ = 6.01 (pSer) / 6.30 (pThr)
* **stickiness**: $\lambda_{pX} = \lambda_X + \Delta\lambda$、$\Delta\lambda = -0.37$
  この値は、11 種のリン酸化 IDR の SAXS / smFRET データに対して
  $\Delta R_g / R_g^{\mathrm{ph}}$ の RMSE を最小化するパラメータスキャンから決められたもので、
  $\lambda_{p\mathrm{Thr}} \approx 0$ に対応します（これより小さくすると λ が負になり
  Ashbaugh–Hatch ポテンシャルが非物理的になるため、ここが下限）。

pTyr も `calvados_ff.py` には入れてありますが、**論文のデータセットに
phospho-Tyr が含まれていないため検証されていません**。使う場合はその前提で。

### 3.3 論文 Fig. 3 の比較モデル

`config.yaml` の `mode` を変えるだけで、論文の対照実験を再現できます。

| mode | 内容 |
|---|---|
| `phospho` | データ駆動モデル（Δλ = −0.37、bead 拡大、q ≈ −2） |
| `unphos` | 脱リン酸化（baseline） |
| `charge` | サイズと λ は Ser/Thr のまま、電荷だけ −2（"charge model"） |
| `mimetic` | pSer → Asp、pThr → Glu（phosphomimetic、電荷 −1） |

`config.yaml` の `systems:` にコメントアウトで用意してあるので、
外せばそのまま 4 系比較になります。論文では、コンパクション変化の大部分は
**電荷の寄与**で説明でき、hydropathy の寄与はより小さいと結論されています。
また phosphomimetic は電荷が −1 しかないため、リン酸化の効果を過小評価します。

---

## 4. 設定 (`config.yaml`)

```yaml
simulation:
  temperature: 298.0      # K
  ionic_strength: 0.15    # M
  pH: 7.4                 # pSer の電荷を決める
  timestep_fs: 10.0
  friction_per_ps: 0.01
  dlambda: -0.37          # 論文の最終値
  box_nm: auto            # auto -> ceil((N-1)*0.38 + 4) = 153 nm
  total_time_ns: 1000     # 1 us / レプリカ
  save_interval_ps: 200   # -> 5000 フレーム / レプリカ
  equil_time_ns: 100      # 解析で捨てる冒頭
  n_replicas: 4           # GPU 台数に合わせる
  max_wall_hours: 7       # これを超えたら中断して自動再キュー (0 で無制限)
```

**変えるときの目安**

* `n_replicas` … 使える GPU 台数に合わせる。系が 2 つなので、GPU 4 枚なら
  `n_replicas: 4` で 8 ジョブ = 2 巡。GPU 8 枚なら 1 巡で終わります。
* `save_interval_ps` … 200 ps は「フレーム間の相関を残しつつ分布を滑らかに描く」
  妥協点。分布のバーを細かくしたければ 100 ps に。誤差評価はブロック平均で
  相関を考慮しているので、細かくしても誤差が不当に小さくはなりません。
* `box_nm` … `auto` は完全伸長鎖が入るサイズ（CALVADOS の慣習）で安全側。
  60 nm 程度まで下げても、この鎖長・この静電カットオフ（4 nm）なら
  自己像との相互作用はまず起きません。

---

## 5. HTCondor での実行

### 5.1 投入

```bash
cd condor
./submit_all.sh              # config の全 system × n_replicas
./submit_all.sh phos         # phos だけ
./submit_all.sh phos unphos 8  # 最後が数字ならレプリカ数の上書き
```

`submit_all.sh` は `config.yaml` を読んで `jobs.txt` を生成し、

```
queue system, replica from jobs.txt
```

で **1 ジョブ = 1 GPU = 1 レプリカ**として投入します。`request_gpus = 1` なので、
空いている GPU から順に走り、足りなければキューで待ちます。

### 5.2 NMRbox 固有の設定

`condor/simulate.sub` は NMRbox の HTCondor プールに合わせてあります。
効いているのは次の点です。

| 設定 | 理由 |
|---|---|
| `executable` を絶対パスで指定 | NMRbox では相対パスが使えない。`submit_all.sh` が `-append` で埋める |
| `should_transfer_files = NO`, `transfer_executable = FALSE` | NMRbox の計算ノードは home を共有しているのでファイル転送が不要 |
| `getenv = True` | 投入時のシェル環境（venv の PATH など）を引き継ぐ |
| `request_cpus = 1` | NMRbox はマルチコアジョブを強く非推奨。1 コアのほうが早く空きが見つかる |
| `request_memory = 4 GB` | 省略すると 2 GB 超で強制終了される。CG MD なので 4 GB で十分 |
| `+Production = True` | NMR ソフトが入った production ノードに限定 |
| `gpus_minimum_memory = 2000MB` | 392 ビーズの単鎖なので GPU メモリはほとんど要らない |
| `on_exit_remove` | 後述の自動分割のため |

**プロジェクトはホームディレクトリ以下に置いてください。** NMRbox の計算ノードは
home を共有していますが、scratch やマシンごとの一時領域はジョブから見えません。
`.venv` も同じ理由でホーム以下（＝プロジェクト直下）に置く必要があります。

**既定は「GPU の質を問わず、とにかく多くのノードにマッチさせる」設定**です。
具体的には次のようにしてあります。

* `gpus_minimum_memory` を指定しない — 392 ビーズの単鎖なので必要な GPU
  メモリは数十 MB 程度。条件を書かないほうがマッチが広がります
* `GPUs_Capability` を指定しない — T4 や V100 といった旧世代も使います
* `+Production = False` — NMR ソフトを一切使わず venv で完結しているので、
  compute-only ノードも動員します
* `request_memory = 2 GB` / `request_cpus = 1` — 小さく要求するほど
  空きスロットが見つかりやすくなります

逆に速い GPU だけを狙いたくなったら、`simulate.sub` の次の行を有効にしてください
（当然、待ち時間は延びます）。

```
requirements        = GPUs_Capability >= 8
gpus_minimum_memory = 4000MB
```

### 異機種プールでの自動フォールバック

条件を緩めるとノードごとにドライバや GPU 世代がばらつきます。`simulate.py` は
プラットフォームが「一覧に出ているか」ではなく**実際に Context を作って
カーネルをコンパイルできるか**で判定し、CUDA → OpenCL → CPU の順に
自動でフォールバックします。PTX バージョン不一致のノードに当たっても
ジョブは落ちず、OpenCL で走り続けます。

どのプラットフォームで走ったかはログの `platform :` 行に出ます。
CPU にフォールバックしていたら極端に遅いので、`monitor.py` の ns/day 列で
すぐ気づけます。

プールの状況は次で確認できます。

```bash
condor_status -const 'TotalGpus > 0'          # GPU を持つマシン
condor_status -af Machine TotalGpus GPUs_Capability -const 'TotalGpus > 0'
condor_q -better <JobID>                      # ジョブが動かない理由
```

### 5.3 8 時間ルールと自動分割

NMRbox は「1 ジョブ 8 時間以内」を推奨しています。1 µs のランは GPU によっては
これを超えるため、**自動で分割する仕組み**を入れてあります。

`config.yaml` の

```yaml
max_wall_hours: 7
```

を超えると `simulate.py` はチェックポイントを保存して **exit 85** で抜けます。
submit ファイルの

```
on_exit_remove = (ExitCode == 0) || (NumJobStarts >= 20)
```

により、そのジョブは自動でキューに戻り、次に走り出したときに
**チェックポイントから続きを実行**します。これが 1 µs に到達するまで繰り返されます。

トラジェクトリの追記は DCD → checkpoint の順で行っているため、
何度中断・再開してもフレームは重複しません（2 回の中断を挟んだ
テストでフレーム数が理論値ぴったりになることを確認済みです）。

自前のマシンで回すなど時間制限が要らない場合は `max_wall_hours: 0` で無制限になります。

### 5.4 GPU の割り当て

HTCondor は `request_gpus` を使うと `CUDA_VISIBLE_DEVICES` または
`_CONDOR_AssignedGPUs` をジョブ環境に設定します。`run_sim.sh` が両方を見て
`CUDA_VISIBLE_DEVICES` に正規化し、`simulate.py` はそれを尊重して
`DeviceIndex=0`（＝割り当てられた 1 枚）を使います。
**複数ジョブが同じ GPU に載ることはありません。**

もし condor 側で GPU が見えていない場合は、実行ノードの設定で
`condor_gpu_discovery` が有効になっているか確認してください:

```bash
condor_status -compact -constraint 'TotalGpus > 0'
condor_gpu_discovery -properties
```

### 5.5 途中で止まっても大丈夫

`simulate.py` は `save_interval_ps` ごとにチェックポイントを書き、
次回起動時に自動で続きから再開します。レポータの順番を
DCD → checkpoint にしてあるので、**再開時にフレームが重複しません**。
`simulate.sub` には `max_retries` と `periodic_release` を入れてあるので、
evict されても自動で復帰します。最初からやり直したいときは `--fresh`。

---

## 6. 進捗の見かた

```bash
python monitor.py            # 1 回表示
python monitor.py --watch    # 5 秒ごとに更新
python monitor.py --watch --condor   # condor_q も一緒に
```

```
job                   progress                          ns    ns/day    ETA   <Rg>    Rg  state
-----------------------------------------------------------------------------------------------
phos/rep0             ██████████░░░░░░░░  38.20%   382.0/1000   4820   3.1h   5.42  5.19  実行中 @nmr01
phos/rep1             ████████░░░░░░░░░░  31.05%   310.5/1000   4655   3.6h   5.38  5.71  実行中 @nmr01
unphos/rep0           ████████████░░░░░░  44.10%   441.0/1000   5010   2.7h   4.96  4.88  実行中 @nmr02
```

* 各ジョブは `runs/<system>_rep<N>/status.json` を更新し続けます
* stdout（condor の `logs/*.out`）にも同じ進捗バーが出ます
* 15 分以上更新が無いジョブは「停止?」と赤く出ます
* `state.log` には OpenMM の StateDataReporter がエネルギー・温度・速度を記録

---

## 7. 解析

```bash
python analyze.py                              # config の compare: に従う
python analyze.py --systems unphos phos --equil-ns 200
```

やっていること:

1. **読み込み** — 各系の全レプリカを読み、冒頭 `equil_time_ns` を捨てて連結。
   周期境界で鎖が切れていた場合は最小イメージ規約でつなぎ直します。
2. **$R_g$** — CALVADOS 流の質量重み付き（N 末端 +2、C 末端 +16 Da）。
3. **誤差** — ブロック平均法。フレーム間の相関を考慮した平均の標準誤差と、
   有効サンプル数 $n_\mathrm{eff}$ を出します。**単純な SD/√N は使っていません**
   （MD の連続フレームは独立ではないため、誤差を大幅に過小評価します）。
4. **$R_{ee}$、内部スケーリング** — $\sqrt{\langle d_{ij}^2\rangle} = R_0 |i-j|^\nu$ を
   $|i-j| > 5$ でフィットして Flory 指数 ν を出します。
5. **構造抽出** — $R_g$ でソートして、**最もコンパクトな上位 5 構造**と
   **最も伸びた上位 5 構造**、および分布の最頻値付近の代表構造を PDB で保存。
   すべて最コンパクト構造に重ね合わせ済みで、まとめて 1 本の
   マルチモデル PDB (`*_extremes_multimodel.pdb`) にも出します。
6. **距離マップ差分** — $\langle d_{ij}\rangle$ のリン酸化 − 非リン酸化。
   赤 = リン酸化で離れた領域、青 = 近づいた領域。リン酸化サイトに細い線が入ります。

7. **主成分分析 (PCA)** — 折り畳みタンパク質と違い IDR には単一の参照構造が
   無いため、Cartesian 座標を重ね合わせる通常の PCA はあまり意味を持ちません。
   代わりに **残基間距離行列に対する PCA**（distance-matrix PCA）を行います。

   * 各フレームの残基対間距離を平坦化したベクトルを特徴量とする
     （392 残基は全対だと ~76,600 次元になるので、既定では
     `pca_residue_stride: 3` で残基を間引いて ~8,500 次元に抑えます）
   * **両系のフレームをまとめて**フィッティングし、共通の主成分空間を作る
     （系ごとに別々に PCA すると、そもそも座標系が違うので比較できません）
   * フィッティングは `pca_max_frames_per_system` でフレームを間引いて
     計算コストを抑えますが、**射影は全フレームに対して行う**ので
     統計量は落ちません
   * `pca.png` の 3 パネル:
     1. PC1-PC2 平面上の散布図 + KDE 等高線（両系を重ねて表示）
     2. 寄与率（棒 = 各 PC、線 = 累積）と PC1-$R_g$ の相関係数
     3. PC1 の loading を残基対の N×N マップとして可視化
        （どの残基間接触が PC1 の動きを支配しているか）
   * IDR では PC1 が概ね全体の広がり（$R_g$）に対応することが多く、
     相関係数 `corr(PC1, Rg)` が出力されます。1 に近ければ
     「PCA も結局 $R_g$ と同じ情報を見ている」ということなので、
     PC2 以降やloading map の方が新しい情報（どこがどう動くか）を持ちます

   `config.yaml` の `analysis.pca: false` で無効化できます。
   `scikit-learn` が無い環境では自動的にスキップされ（他の解析は実行されます）、
   `pip install scikit-learn`（`requirements.txt` に含まれています）で有効になります。

8. **分子内相互作用** — 「どれくらい広がっているか」(Rg) の一段階先、
   「具体的にどの残基がどれと・どうやって相互作用しているか」を調べます。
   simulate.py が使っているのと**厳密に同じ力場の式**（Ashbaugh-Hatch と
   Debye-Hückel）を numpy で再実装しており、実際に OpenMM が計算する
   エネルギーと突き合わせて一致することを検証済みです（誤差 <0.01%）。

   * **接触確率マップ** — 残基対 $(i,j)$ について、距離が
     `contact_cutoff_factor × (σ_i+σ_j)/2` 未満になっているフレームの
     割合。既定の `contact_cutoff_factor: 1.5` は「ビーズが実質的に
     触れ合っている」とみなせる目安の距離です。
   * **電荷タイプ別の接触エンリッチメント** — 残基を
     `phospho` / `acidic` (D, E) / `basic` (K, R) / `his` / `other` に分類し、
     タイプの組み合わせごとに接触確率を比較します。
     **単純な「全ペア平均」を期待値にすると、配列上で近い残基同士は
     鎖の連結性だけで接触しやすいため、たまたま近くに集まっている
     電荷タイプを過大評価してしまいます**（今回の配列は pSer が
     3–12 残基間隔でクラスターを作っているため、この罠に典型的に
     引っかかります）。そこで、**同じ配列間隔 $|i-j|$ を持つ全ペアの
     平均接触確率を期待値とする補正**をかけています
     (`enrichment` 列。補正前の単純版は `enrichment_naive` 列に残してあります)。
     この補正版で見ると、この配列では
     `basic-phospho`（K/R–pSer の塩橋）が突出してエンリッチ（3 倍以上）、
     `phospho-phospho`（pSer 同士）は強く忌避される（0.1 倍程度）
     という、論文が主張する静電機構と整合する結果が得られます
     （数値は配列依存であり、実際の 1 µs ランで確認してください）。
   * **相互作用エネルギーの分解** — Ashbaugh-Hatch（疎水性/排除体積、
     `E_AH`）と Debye-Hückel（塩遮蔽静電、`E_DH`）を独立に積算します。
     `E_DH` の符号は正味で引力的か斥力的かを直接示す量です。
     残基ごとのエネルギー寄与（ペアのエネルギーを両端で折半）も
     `residue_energy_*.csv` に出すので、「配列のどこが効いているか」を
     残基レベルで確認できます。
   * **形状記述子** — 慣性楕円体の固有値から非対称度 (asphericity) と
     形状異方性 $\kappa^2$（0=球形、1=棒状）を計算します。相互作用の
     強さが実際にどんな「形」を作っているかを見る指標です。
   * **配列電荷デコレーション (SCD)** — シミュレーションとは独立に、
     配列だけから計算できる静的指標です。負に大きいほど正負電荷が
     配列上でブロック的に分離しており、長距離の静電引力が働きやすい
     配列であることを示します（Sawle & Ghosh 2015）。

   計算コストの重い部分（残基対ごとの距離計算）は `analysis.stride_map`
   と同じ間引きフレームに対して行われます。`config.yaml` の
   `analysis.interactions: false` で無効化できます。

### 構造の見かた

```bash
pymol analysis/structures/phos_extremes_multimodel.pdb
# または
vmd analysis/structures/phos_compact01_Rg*.pdb
```

粗視化なので Cα 相当の 1 bead/残基です。リボン表示にはならないので、
PyMOL なら `show spheres` や `set ribbon_trace_atoms, 1; show ribbon` が見やすいです。
全原子構造が必要になったら [cg2all](https://github.com/huhlim/cg2all) で
後から変換できますが、今回は「粗視化のままで高速に」という方針なので入れていません。

---

## 8. この系について予想されること

配列の電荷組成は次の通りです（pH 7.4、末端の電荷込み）:

| | 正味電荷 | NCPR | FCR | ⟨λ⟩ |
|---|---|---|---|---|
| 非リン酸化 | −8.66 | −0.022 | 0.177 | 0.438 |
| 12× pSer | **−32.19** | **−0.082** | 0.237 | 0.427 |

12 箇所のリン酸化で正味電荷が約 −23.5 動きます（1 サイトあたり −1.96）。
**元々酸性寄りの配列に負電荷を足す**形なので、論文の議論
（既に酸性アミノ酸に富む IDR では静電反発の増大で伸長する）からは
**$R_g$ が増大する方向**が予想されます。stickiness の低下（λ_pSer = 0.09）も
同じ向きに働きます。実際にどうなるかは走らせて確認してください。

リン酸化サイトが 168–180 / 213–225 / 268–276 の 3 クラスタに固まっているので、
距離マップ差分でクラスタ間・クラスタと正電荷領域（K, R が多い 240–260 付近）の
相対距離がどう変わるかを見ると、局所的な効果が読み取れます。

---

## 9. トラブルシュート

**`CUDA` プラットフォームが見つからない**

```bash
python -c "import openmm; print(openmm.Platform.getPlatformByName('CUDA'))"
```

で落ちる場合、`openmm` を extra 無しで入れています。

```bash
pip install "openmm[cuda12]"     # ドライバが CUDA 12.x 対応なら
pip install "openmm[cuda13]"     # 13.x なら
```

`simulate.py` は CUDA → OpenCL → CPU の順に**黙ってフォールバックする**ので、
気づかず CPU で走ってしまうことがあります。ログの `platform :` 行を必ず確認してください。
`scripts/benchmark.py` を先に流せば一目で分かります。

**`CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)` で落ちる**

`Platforms` に CUDA が出ているのに Context を作る段階で落ちる場合、
NVRTC（CUDA の実行時コンパイラ）がドライバより新しすぎます。
pip は依存解決で最新の `nvidia-cuda-nvrtc-cu12` を入れてしまうため、
古めのドライバのマシンでは PTX を読めずに落ちます。

`nvidia-smi` の右上の `CUDA Version:` を見て、その次のマイナー版未満に固定します。

```bash
pip install "nvidia-cuda-nvrtc-cu12<12.2"    # ドライバが CUDA 12.1 のとき
pip install "nvidia-cuda-nvrtc-cu12<12.5"    # ドライバが CUDA 12.4 のとき
```

ドライバのバージョンと CUDA 版の対応の目安: 525 → 12.0、530 → 12.1、
535 → 12.2、550 → 12.4、560 → 12.6。

`scripts/setup_env.sh` はこの固定を自動でやり、最後に実際にカーネルを
コンパイルできるかまで確認します。それでも直らない場合は OpenCL でも動きます
（T4 で 2〜3 割遅い程度）。

```bash
python simulate.py --platform OpenCL --system phos --replica 0
```

**condor ジョブがすぐ hold になる**

`condor_q -hold` で理由を見てください。よくあるのは venv が渡っていないケースです。
`run_sim.sh` は `<project>/.venv` を自動で activate しますが、別の場所に作った場合は

```
environment = "VENV_DIR=/path/to/venv"
```

を `simulate.sub` に足すか、`VENV_DIR` を export してから `condor_submit` してください
（`getenv = True` なので投入時のシェルの環境が引き継がれます）。

**図の日本語が豆腐になる**

日本語フォントが無い環境では自動で英語ラベルに切り替わります（図自体は問題なく出ます）。
日本語にしたい場合は次のどちらかを:

```bash
pip install japanize-matplotlib        # root 権限が無くてもよい
sudo apt install fonts-noto-cjk        # システムに入れる場合
```

**フレーム数が足りないと言われる**

`equil_time_ns` がトラジェクトリ長より長い可能性があります。
`--equil-ns 0` を付けて確認してください。

---

## 10. 引用

* Rauh, A.S., Hedemark, G.S., Tesei, G., Lindorff-Larsen, K. (2026)
  *Biophys. J.* **125**, 396–405.
* Tesei, G., Lindorff-Larsen, K. (2023) *Open Res. Europe* **2**, 94. (CALVADOS 2)
* Tesei, G. et al. (2021) *PNAS* **118**, e2111696118. (CALVADOS)
* Eastman, P. et al. (2017) *PLoS Comput. Biol.* **13**, e1005659. (OpenMM)

公式実装は
[KULL-Centre/CALVADOS](https://github.com/KULL-Centre/CALVADOS) と
[KULL-Centre/_2025_rauh_phosphorylation](https://github.com/KULL-Centre/_2025_rauh_phosphorylation)
にあります。本コードはそれらとは独立に、論文記載の式とパラメータから実装したものです。
論文の値（λ_pSer = 0.09、λ_pThr ≈ 0）が再現されることは確認済みです。
