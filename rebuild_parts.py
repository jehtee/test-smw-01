"""
rebuild_parts.py

Recovers the appraisal parts of an appraisal run that failed later on (for
example on ordinary text), so a rerun need not write them again. judged.jsonl
holds every answer the judge saw, in order: per event, the appraisal batches
then the story batches, then ordinary text. An appraisal was kept exactly when
the judge said drop=false, and collect() kept the first --n of those. Each one
answers a fresh neutral question in training, so nothing else is needed.

Story-only parts can't be recovered (judged.jsonl doesn't keep their
questions); the rerun writes those again, which is quick.

    python3 rebuild_parts.py r2_data/Qwen3.8-27B-8bit/s0 --seed 0
"""
import argparse
import itertools
import json
import os
import random

from appraise import NEUTRAL_ASKS
from battery_r2 import EVENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="the appraisal run's folder (holds judged.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--min", type=int, default=40)
    args = ap.parse_args()

    with open(os.path.join(args.out, "judged.jsonl")) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    runs = [list(g) for _, g in itertools.groupby(rows, key=lambda r: r["kind"])]
    apps = [g for g in runs if g[0]["kind"] == "appraisal"]
    if len(apps) != len(EVENTS):
        raise SystemExit(f"expected {len(EVENTS)} runs of appraisals in judged.jsonl, found {len(apps)}")

    rng = random.Random(args.seed)
    cache = os.path.join(args.out, "parts")
    os.makedirs(cache, exist_ok=True)
    for eid, g in zip(EVENTS, apps):
        kept = [r["text"] for r in g if r["drop"] is False][:args.n]
        path = os.path.join(cache, f"appraisal_{eid}.jsonl")
        if len(kept) < args.min:
            print(f"appraisal_{eid}: {len(kept)}, too few, not saved")
            continue
        with open(path, "w") as fh:
            for a in kept:
                fh.write(json.dumps([rng.choice(NEUTRAL_ASKS), a]) + "\n")
        print(f"appraisal_{eid}: {len(kept)} saved")


if __name__ == "__main__":
    main()
