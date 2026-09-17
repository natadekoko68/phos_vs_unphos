#!/bin/bash
# ==========================================================================
#  HTCondor を使わず、1 台のマシンの複数 GPU に直接ジョブを配る。
#  (condor が使えないとき / まず動作確認したいときの簡易版)
#
#    ./scripts/run_local_multigpu.sh                 # config の全 system
#    ./scripts/run_local_multigpu.sh phos unphos     # 系を指定
#    NREP=2 ./scripts/run_local_multigpu.sh          # レプリカ数を上書き
#
#  ジョブは nohup でバックグラウンドに投げる。進捗は
#    python monitor.py --watch
# ==========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs logs

# --- venv の有効化 (conda 不要) ---
VENV_DIR="${VENV_DIR:-$(pwd)/.venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
elif [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "warn: venv が見つかりません。先に ./scripts/setup_env.sh を実行してください。"
fi

NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$NGPU" -eq 0 ]; then
    echo "GPU が見つかりません。CPU で 1 本だけ走らせるなら:"
    echo "  python simulate.py --system phos --replica 0 --platform CPU"
    exit 1
fi
echo "検出した GPU: $NGPU 枚"

# --- ジョブリストを作る ---
mapfile -t JOBS < <(python - "$@" << 'PY'
import sys, yaml, os
cfg = yaml.safe_load(open("config.yaml"))
nrep = int(os.environ.get("NREP") or cfg["simulation"].get("n_replicas", 4))
systems = [a for a in sys.argv[1:]] or list(cfg["systems"])
for s in systems:
    for r in range(nrep):
        print(f"{s} {r}")
PY
)

echo "投入するジョブ: ${#JOBS[@]} 本"
printf '  %s\n' "${JOBS[@]}"

i=0
for job in "${JOBS[@]}"; do
    read -r SYS REP <<< "$job"
    GPU=$(( i % NGPU ))
    LOG="logs/${SYS}_rep${REP}.out"
    echo "  -> GPU $GPU : $SYS rep$REP  (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u simulate.py \
        --config config.yaml --system "$SYS" --replica "$REP" \
        > "$LOG" 2>&1 &
    echo $! > "runs/${SYS}_rep${REP}.pid"
    i=$(( i + 1 ))
    sleep 2       # 同時に CUDA コンテキストを作らせない
done

echo
echo "全ジョブをバックグラウンドで開始しました。"
echo "進捗:  python monitor.py --watch"
echo "停止:  pkill -f 'simulate.py --config'"
wait
