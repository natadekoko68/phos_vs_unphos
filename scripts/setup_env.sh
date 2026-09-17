#!/bin/bash
# ==========================================================================
#  venv + pip で環境を作る (conda は使いません)
#
#    ./scripts/setup_env.sh              # .venv を作って全部入れる
#    ./scripts/setup_env.sh /path/to/env # 場所を指定
#    CUDA_EXTRA=cuda13 ./scripts/setup_env.sh   # CUDA の版を明示
#    CUDA_EXTRA=none   ./scripts/setup_env.sh   # CPU のみ (GPU 無しのマシン)
#
#  NVIDIA ドライバさえ入っていれば、CUDA Toolkit のインストールは不要です。
#  OpenMM の CUDA プラグイン (libOpenMMCUDA.so) は pip ホイールに同梱されます。
# ==========================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_DIR="$(pwd)"
VENV="${1:-$PROJECT_DIR/.venv}"

echo "=============================================================="
echo " プロジェクト : $PROJECT_DIR"
echo " venv         : $VENV"
echo "=============================================================="

# --- Python の確認 ---------------------------------------------------------
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)'; then
    echo "エラー: Python 3.9 以上が必要です (今: $("$PY" --version 2>&1))"
    exit 1
fi
echo "Python : $("$PY" --version 2>&1)  ($(command -v "$PY"))"

# --- CUDA のメジャー版を判定 -----------------------------------------------
# ドライバが対応する CUDA のバージョン (例: "12.1") を取得する
driver_cuda_version() {
    command -v nvidia-smi > /dev/null 2>&1 || return 1
    nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]\+\.[0-9]\+\).*/\1/p' | head -1
}
DRIVER_CUDA="$(driver_cuda_version || true)"
CUDA_MAJOR="${DRIVER_CUDA%%.*}"
CUDA_MINOR="${DRIVER_CUDA##*.}"

detect_cuda_extra() {
    if [ -n "${CUDA_EXTRA:-}" ]; then echo "$CUDA_EXTRA"; return; fi
    if [ -z "$DRIVER_CUDA" ]; then echo "none"; return; fi
    case "$CUDA_MAJOR" in
        13) echo "cuda13" ;;
        12) echo "cuda12" ;;
        11) echo "none"   ;;   # OpenMM 8.x の pip ホイールは CUDA 12 以降のみ
        *)  echo "cuda12" ;;   # 判定できなければ一番一般的な 12 を試す
    esac
}
EXTRA="$(detect_cuda_extra)"

if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
else
    echo "GPU   : nvidia-smi が見つかりません"
fi
echo "OpenMM extra : ${EXTRA}"
if [ "$EXTRA" = "none" ]; then
    echo "  警告: CUDA 版 OpenMM を入れません。CPU / OpenCL のみになります。"
    echo "        GPU があるのにこうなる場合は CUDA_EXTRA=cuda12 を指定してください。"
fi

# --- venv ------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
    echo "venv を作成しました"
else
    echo "既存の venv を使います"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel > /dev/null
echo "--- 依存パッケージのインストール ---"
pip install -r requirements.txt

if [ "$EXTRA" != "none" ]; then
    echo "--- CUDA プラグイン (openmm[$EXTRA]) ---"
    if ! pip install "openmm[$EXTRA]"; then
        echo "警告: openmm[$EXTRA] の導入に失敗しました。CPU/OpenCL で動作します。"
    fi

    # ------------------------------------------------------------------
    # NVRTC をドライバに合わせて固定する
    #
    # OpenMM は CUDA カーネルを NVRTC で実行時コンパイルする。pip の依存解決は
    # 最新の nvidia-cuda-nvrtc-cu12 (例 12.9) を入れてしまうため、ドライバが
    # 古いと生成された PTX を読めず
    #     CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)
    # で落ちる。ドライバが対応する CUDA 版以下に固定して回避する。
    # 例: ドライバ 530.30.02 -> CUDA 12.1 -> nvrtc < 12.2
    # ------------------------------------------------------------------
    if [ -n "$DRIVER_CUDA" ] && [ "$CUDA_MAJOR" = "12" ]; then
        NEXT_MINOR=$(( CUDA_MINOR + 1 ))
        PIN="nvidia-cuda-nvrtc-cu12<${CUDA_MAJOR}.${NEXT_MINOR}"
        echo "--- NVRTC をドライバ (CUDA $DRIVER_CUDA) に合わせて固定 ---"
        echo "    $PIN"
        if pip install "$PIN"; then
            pip show nvidia-cuda-nvrtc-cu12 2>/dev/null | sed -n 's/^Version: /    -> nvrtc /p'
        else
            echo "    警告: NVRTC の固定に失敗しました。"
            echo "    CUDA_ERROR_UNSUPPORTED_PTX_VERSION が出る場合は手動で:"
            echo "        pip install \"$PIN\""
        fi
    fi
fi

# --- 動作確認 --------------------------------------------------------------
echo
echo "=============================================================="
python - << 'PY'
import openmm
print("OpenMM :", openmm.version.version)
names = [openmm.Platform.getPlatform(i).getName()
         for i in range(openmm.Platform.getNumPlatforms())]
print("Platforms :", ", ".join(names))
if "CUDA" in names:
    print("  -> CUDA が使えます")
elif "OpenCL" in names:
    print("  -> CUDA なし。OpenCL で動きます (やや遅い)")
else:
    print("  -> GPU プラグインがありません。CPU のみです")
import mdtraj, numpy, scipy, pandas, matplotlib, yaml
print("mdtraj :", mdtraj.__version__)

# CUDA は「一覧に出る」だけでは不十分で、カーネルの実行時コンパイルが
# 通るかまで確かめないと PTX バージョン不一致に気づけない
if "CUDA" in names:
    try:
        from openmm import unit
        sysm = openmm.System()
        sysm.addParticle(1.0 * unit.amu)
        sysm.addParticle(1.0 * unit.amu)
        f = openmm.CustomNonbondedForce("r^2")
        f.addParticle([]); f.addParticle([])
        sysm.addForce(f)
        ctx = openmm.Context(sysm, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("CUDA"))
        ctx.setPositions([(0, 0, 0), (0.1, 0, 0)])
        ctx.getState(getEnergy=True).getPotentialEnergy()
        print("CUDA カーネルのコンパイル : OK")
    except Exception as e:
        print("CUDA カーネルのコンパイル : 失敗")
        print("   ", e)
        if "PTX" in str(e):
            print("    -> NVRTC がドライバより新しすぎます。次を実行してください:")
            print('       pip install "nvidia-cuda-nvrtc-cu12<12.2"   # 数字はドライバの CUDA 版+0.1')
        print("    -> 直らない場合は OpenCL で走らせられます "
              "(--platform OpenCL)")
PY
echo "=============================================================="
echo
echo "使うときは毎回:"
echo "    source $VENV/bin/activate"
echo
echo "次のステップ:"
echo "    python scripts/benchmark.py --all-gpus"
