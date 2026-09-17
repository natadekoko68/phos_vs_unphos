#!/bin/bash
# ==========================================================================
#  config.yaml を読んで jobs.txt を作り、HTCondor に投入する。
#  (NMRbox の HTCondor プール向け)
#
#    ./submit_all.sh                # config の全 system × n_replicas
#    ./submit_all.sh phos           # phos だけ
#    ./submit_all.sh phos unphos 8  # 最後が数値ならレプリカ数の上書き
#    DRY=1 ./submit_all.sh          # jobs.txt を作るだけで投入しない
# ==========================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
CONDOR_DIR="$(pwd)"
PROJECT_DIR="$(cd .. && pwd)"
mkdir -p logs "$PROJECT_DIR/runs"

# --- 事前チェック -----------------------------------------------------------
if ! command -v condor_submit > /dev/null 2>&1; then
    echo "エラー: condor_submit が見つかりません。"
    echo "  NMRbox のマシンにログインしているか確認してください。"
    exit 1
fi

case "$PROJECT_DIR" in
    /home/*|/mnt/*) : ;;
    *) echo "警告: $PROJECT_DIR がホーム以下にありません。"
       echo "      NMRbox の計算ノードは home を共有していますが、"
       echo "      scratch やローカル一時領域だとジョブから見えません。" ;;
esac

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "警告: venv が見つかりません ($VENV_DIR)"
    echo "      先に $PROJECT_DIR/scripts/setup_env.sh を実行してください。"
fi

chmod +x run_sim.sh

# --- 引数のパース -----------------------------------------------------------
NREP_OVERRIDE=""
SYSTEMS=()
for a in "$@"; do
    if [[ "$a" =~ ^[0-9]+$ ]]; then NREP_OVERRIDE="$a"; else SYSTEMS+=("$a"); fi
done

PY="$([ -x "$VENV_DIR/bin/python" ] && echo "$VENV_DIR/bin/python" || echo python3)"

"$PY" - "$PROJECT_DIR" "${NREP_OVERRIDE:-}" "${SYSTEMS[@]:-}" << 'PY' > jobs.txt
import sys, yaml, pathlib
proj = pathlib.Path(sys.argv[1])
nrep_override = sys.argv[2]
wanted = [a for a in sys.argv[3:] if a]
cfg = yaml.safe_load(open(proj / "config.yaml"))
nrep = int(nrep_override) if nrep_override else int(cfg["simulation"].get("n_replicas", 4))
systems = wanted or list(cfg["systems"])
for s in systems:
    if s not in cfg["systems"]:
        sys.exit(f"unknown system: {s}")
    for r in range(nrep):
        print(f"{s}, {r}")
PY

echo "--- jobs.txt ---"
cat jobs.txt
echo "----------------"
N=$(wc -l < jobs.txt)
echo "$N ジョブ (それぞれ GPU 1 枚) を投入します"
echo "  project : $PROJECT_DIR"
echo "  venv    : $VENV_DIR"

if [ -n "${DRY:-}" ]; then
    echo "(DRY=1 のため投入しません)"
    exit 0
fi

# NMRbox では executable に絶対パスが必要。venv の場所もジョブに渡す。
condor_submit simulate.sub \
    -append "executable = $CONDOR_DIR/run_sim.sh" \
    -append "environment = \"VENV_DIR=$VENV_DIR\""

echo
echo "--- プールの GPU 状況 ---"
condor_status -const 'TotalGpus > 0' \
    -af Machine TotalGpus GPUs_Capability GPUs_GlobalMemoryMb 2>/dev/null \
    | sort | head -30 || echo "(取得できませんでした)"
echo
echo "キュー確認 : condor_q            (詳細は condor_q -nobatch)"
echo "待ち理由   : condor_q -better <JobID>"
echo "進捗       : $PY $PROJECT_DIR/monitor.py --watch --runs $PROJECT_DIR/runs"
echo "ログ       : tail -f $CONDOR_DIR/logs/*.out"
echo "取り消し   : condor_rm <JobID>"
