#!/bin/bash
# ==========================================================================
# HTCondor から呼ばれる実行ラッパー
#   引数:  $1 = system 名 (unphos / phos ...)
#          $2 = replica 番号
#
# Python 環境は venv (conda 不要)。探索順は
#   1. 環境変数 VENV_DIR
#   2. <project>/.venv
#   3. 既に activate 済みの環境 ($VIRTUAL_ENV) / システムの python3
# ==========================================================================
set -euo pipefail

SYSTEM="${1:?system 名が必要です}"
REPLICA="${2:?replica 番号が必要です}"

# --- プロジェクトのルート (このスクリプトの 1 つ上) ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# --- venv の有効化 ---------------------------------------------------------
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "info: 既に activate 済みの venv ($VIRTUAL_ENV) を使います"
else
    echo "warn: venv が見つかりません ($VENV_DIR)。システムの python3 で続行します。"
    echo "      作るには: ./scripts/setup_env.sh"
fi

# --- GPU の割り当て ---------------------------------------------------------
# HTCondor は request_gpus を使うと CUDA_VISIBLE_DEVICES か
# _CONDOR_AssignedGPUs (例: "CUDA0") を環境変数に設定する。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$(echo "$_CONDOR_AssignedGPUs" | sed 's/CUDA//g; s/GPU-//g')"
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "warn: GPU が割り当てられていません。OpenMM の自動選択に任せます。"
fi

echo "===================================================================="
echo " host                : $(hostname)"
echo " date                : $(date)"
echo " project             : $PROJECT_DIR"
echo " system / replica    : $SYSTEM / $REPLICA"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " _CONDOR_AssignedGPUs: ${_CONDOR_AssignedGPUs:-<unset>}"
echo " python              : $(command -v python) ($(python --version 2>&1))"
echo "===================================================================="
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu \
           --format=csv 2>/dev/null || echo "(nvidia-smi なし)"
echo "===================================================================="

export OPENMM_CPU_THREADS="${OMP_NUM_THREADS:-1}"

exec python -u simulate.py \
    --config config.yaml \
    --system "$SYSTEM" \
    --replica "$REPLICA"
