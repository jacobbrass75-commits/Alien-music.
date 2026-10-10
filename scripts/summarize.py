"""Compare runs: averages over the last N iterations of each log.jsonl under a folder.

  python scripts/summarize.py runs/composer/tune1
  python scripts/summarize.py runs --last 100
"""
import argparse
import glob
import json
import os

COLUMNS = ["return", "entropy", "surprise", "lp", "tension", "distinct_chords", "distinct_events",
           "density", "isochrony", "listener_acc", "energy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="runs")
    p.add_argument("--last", type=int, default=50)
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "**", "log.jsonl"), recursive=True)):
        lines = [json.loads(l) for l in open(path) if l.strip()]
        if not lines:
            continue
        tail = lines[-args.last:]
        avg = {k: sum(r[k] for r in tail) / len(tail) for k in COLUMNS if k in tail[0]}
        rows.append((os.path.relpath(os.path.dirname(path), args.root), lines[-1]["iter"], avg))
    if not rows:
        print("no runs found")
        return

    cols = [c for c in COLUMNS if any(c in r[2] for r in rows)]
    width = max(len(r[0]) for r in rows)
    print(f"{'run':<{width}}  {'iter':>6}  " + "  ".join(f"{c[:9]:>9}" for c in cols) + "  flag")
    for name, it, avg in rows:
        flags = []
        if avg.get("entropy", 1) < 0.1:
            flags.append("COLLAPSED")
        if "surprise" in avg and not 0.2 <= avg["surprise"] <= 0.5:
            flags.append("surprise-off-band")
        cells = "  ".join(f"{avg[c]:>9.3f}" if c in avg else f"{'-':>9}" for c in cols)
        print(f"{name:<{width}}  {it:>6}  {cells}  {' '.join(flags)}")


if __name__ == "__main__":
    main()
