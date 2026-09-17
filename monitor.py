#!/usr/bin/env python
"""
monitor.py
==========
走っている / 走り終わった全レプリカの進捗を一覧表示する。

    python monitor.py                 # 1 回表示
    python monitor.py --watch         # 5 秒ごとに更新 (Ctrl-C で終了)
    python monitor.py --watch -n 15   # 15 秒ごと
    python monitor.py --condor        # condor_q の結果も一緒に出す
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

BAR = 28
GREEN, YELLOW, RED, GREY, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[90m", "\033[1m", "\033[0m")


def collect(root: Path):
    rows = []
    for st in sorted(root.glob("*/status.json")):
        try:
            d = json.loads(st.read_text())
        except Exception:
            continue
        d["dir"] = st.parent.name
        d["age_s"] = time.time() - st.stat().st_mtime
        rows.append(d)
    return rows


def fmt(rows, color=True):
    def c(s, col):
        return f"{col}{s}{RESET}" if color else s

    if not rows:
        return "まだステータスファイルがありません (ジョブが始まっていない?)\n"

    out = []
    head = (f"{'job':<22}{'progress':<{BAR + 10}}{'ns':>17}"
            f"{'ns/day':>10}{'ETA':>9}{'<Rg>':>8}{'Rg':>7}  state")
    out.append(c(head, BOLD))
    out.append("-" * len(head))

    tot_pct, n_run, n_done, n_stale = 0.0, 0, 0, 0
    for d in rows:
        pct = d.get("percent", 0.0)
        tot_pct += pct
        filled = int(BAR * pct / 100)
        bar = "█" * filled + "░" * (BAR - filled)

        if d.get("finished"):
            state, col = "完了", GREEN
            n_done += 1
        elif d["age_s"] > 900:                      # 15 分更新なし
            state, col = f"停止? ({d['age_s'] / 60:.0f}分前)", RED
            n_stale += 1
        else:
            state, col = f"実行中 @{d.get('host', '?')}", YELLOW
            n_run += 1

        out.append(
            f"{d.get('label', d['dir']):<22}"
            f"{c(bar, col)} {pct:6.2f}%  "
            f"{d.get('time_ns', 0):8.1f}/{d.get('total_ns', 0):<7g}"
            f"{d.get('ns_per_day', 0):10.0f}"
            f"{d.get('eta_hours', 0):8.1f}h"
            f"{d.get('rg_running_mean_nm', 0):8.2f}"
            f"{d.get('rg_now_nm', 0):7.2f}  {c(state, col)}")

    out.append("-" * len(head))
    out.append(f"合計 {len(rows)} ジョブ:  完了 {n_done} / 実行中 {n_run} / "
               f"停止? {n_stale}    全体進捗 {tot_pct / len(rows):.1f} %")
    return "\n".join(out) + "\n"


def condor_status():
    if not shutil.which("condor_q"):
        return ""
    try:
        r = subprocess.run(["condor_q", "-nobatch"], capture_output=True,
                           text=True, timeout=15)
        return "\n--- condor_q ---\n" + r.stdout
    except Exception as e:
        return f"\n(condor_q 失敗: {e})\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs", help="出力ディレクトリ")
    p.add_argument("--watch", action="store_true")
    p.add_argument("-n", "--interval", type=float, default=5.0)
    p.add_argument("--condor", action="store_true")
    p.add_argument("--no-color", action="store_true")
    a = p.parse_args()

    root = Path(a.runs)
    while True:
        text = fmt(collect(root), color=not a.no_color)
        if a.condor:
            text += condor_status()
        if a.watch:
            os.system("clear" if os.name != "nt" else "cls")
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"  ({root.resolve()})\n")
        print(text, flush=True)
        if not a.watch:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
