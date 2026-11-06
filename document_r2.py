"""
document_r2.py

The record of a finished round-2 run, written next to its results so they can
be mined later without the machine that made them. Into results_r2/<model>/record/:

  README.md          what ran, when, where, with what; every file and its fields
  texts.json         what Homer was told and asked: events, E6, repairs, rearing
                     facts, recall questions and keys, comprehension checks, frames
  items.jsonl        every situation, with its group, scale, predicted direction
  scores.csv         scores.jsonl joined with each item's group and prediction
  recall.csv         recall.jsonl, one row per answer
  conditions.tsv     every (label, seed): rows scored, recall answers, keys recalled
  comprehension.tsv  every comprehension check Homer answered before forgetting
  balance.tsv        each appraisal step's N, judge and answer-token budget
  inventory.tsv      every data and adapter file: size and sha256 (adapters stay on the run machine)
  environment.txt    machine, OS, Python packages, model snapshots, code hashes
  logs/              the run's logs

Reads only; writes only record/. Run from the round-2 folder on the machine that ran it:
    python document_r2.py --model Qwen3.8-27B-4bit --log ~/lab-jobs/r2full/log --log archive/local/*.log
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime

from score_r2 import read_jsonl


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"({e})"


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def texts():
    import battery_r2 as b
    import facts_r2 as f
    out = dict(events=f.EVENTS, scope=b.SCOPE, E6=b.E6,
               repairs={k: f.build(r) for k, r in f.REPAIRS.items()},
               rearing_homer=f.REARING, rearing_herb=f.REARING_HERB,
               recall_questions=b.RECALL_QUESTIONS, recall_keys=b.RECALL_KEYS,
               comprehension=b.COMPREHENSION, comprehension_pass=b.COMPREHENSION_PASS)
    try:
        from llm_test import FRAMES
        out["frames"] = [dict(system=s, user=u) for s, u in FRAMES]
    except Exception as e:
        out["frames"] = f"(not loaded: {e})"
    return out


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def log_span(paths):
    """First and last timestamped '===' line over every log, and every '!!!' line."""
    stamps, problems = [], []
    for name, p in paths:
        with open(p, errors="replace") as fh:
            for line in fh:
                if line.startswith("!!!"):
                    problems.append(f"{name}: {line.strip()}")
                if line.startswith("=== "):
                    stamps.append(f"{name}: {line.strip()}")
    return stamps, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3.8-27B-4bit")
    ap.add_argument("--log", action="append", default=[], help="a log of this run (repeatable)")
    ap.add_argument("--note", action="append", default=[], help="a provenance note for README (repeatable)")
    ap.add_argument("--models", default="mlx-community/Qwen3.8-27B-4bit,mlx-community/gemma-4-31b-it-4bit,"
                                        "mlx-community/Qwen3.5-122B-A10B-4bit",
                    help="the models whose snapshots to record")
    args = ap.parse_args()

    T = args.model
    R, D, A = f"results_r2/{T}", f"r2_data/{T}", f"adapters_r2/{T}"
    out = os.path.join(R, "record")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    import battery_r2 as b

    # What Homer was told and asked
    with open(os.path.join(out, "texts.json"), "w") as fh:
        json.dump(texts(), fh, indent=1)
    items = b.ITEMS + b.ILSE_ITEMS
    with open(os.path.join(out, "items.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(s) + "\n" for s in items))
    by_id = {s["id"]: s for s in items}

    # Scores and recall, flat
    scores = read_jsonl(os.path.join(R, "scores.jsonl"))
    recall = read_jsonl(os.path.join(R, "recall.jsonl"))
    with open(os.path.join(out, "scores.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "method", "dose", "seed", "item", "group", "pred", "clean", "frame", "ev", "mass"])
        for r in scores:
            s = by_id.get(r["item"], {})
            w.writerow([r["label"], r["method"], r["dose"], r["seed"], r["item"], s.get("group"), s.get("pred"),
                        s.get("clean"), r["frame"], r["ev"], r["mass"]])
    keys = b.RECALL_KEYS
    def hits(answer):
        a = answer.lower()
        return sorted(e for e, ks in keys.items() if any(k in a for k in ks))
    with open(os.path.join(out, "recall.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "method", "dose", "seed", "kind", "question", "answer", "keys_hit"])
        for r in recall:
            w.writerow([r["label"], r["method"], r["dose"], r["seed"], r["kind"], r["question"], r["answer"],
                        " ".join(hits(r["answer"]))])

    # Every condition, with what was scored
    n_sc, n_rc, n_hit = Counter(), Counter(), defaultdict(Counter)
    for r in scores:
        n_sc[(r["label"], r["seed"])] += 1
    for r in recall:
        k = (r["label"], r["seed"])
        n_rc[k] += 1
        for e in hits(r["answer"]):
            n_hit[k][e] += 1
    conds = sorted(set(n_sc) | set(n_rc), key=lambda k: (k[1], k[0]))
    write_tsv(os.path.join(out, "conditions.tsv"), ["label", "seed", "score_rows", "recall_answers", "keys_recalled"],
              [[l, s, n_sc[(l, s)], n_rc[(l, s)], " ".join(f"{e}:{n}" for e, n in sorted(n_hit[(l, s)].items()))]
               for l, s in conds])

    # Comprehension and balance, from every appraisal step
    comp, bal = [], []
    for p in sorted(glob.glob(f"{D}/s*/**/comprehension.jsonl", recursive=True) + glob.glob(f"{D}/s*/comprehension.jsonl")):
        step = os.path.relpath(os.path.dirname(p), D)
        for r in read_jsonl(p):
            comp.append([step, r.get("event"), r.get("sample"), r.get("passed"), r.get("answer", "").replace("\n", " ")])
    for p in sorted(set(glob.glob(f"{D}/s*/balance.json") + glob.glob(f"{D}/s*/*/balance.json"))):
        with open(p) as fh:
            x = json.load(fh)
        bal.append([os.path.relpath(os.path.dirname(p), D), x.get("N"), x.get("judge"), x.get("train_examples"),
                    json.dumps(x.get("answer_tokens")), " ".join(x.get("off_by_over_20pct", []))])
    write_tsv(os.path.join(out, "comprehension.tsv"), ["step", "event", "sample", "passed", "answer"],
              [list(r) for r in dict.fromkeys(tuple(r) for r in comp)])
    write_tsv(os.path.join(out, "balance.tsv"),
              ["step", "N", "judge", "train_examples", "answer_tokens", "off_by_over_20pct"], bal)

    # Every file the run made, and the code it ran
    inv = []
    for top in (D, "r2_data/rearing", "r2_data/rearing_herb", A, R):
        for dp, _, fs in os.walk(top):
            if os.path.relpath(dp, R).startswith("record"):
                continue
            for f in sorted(fs):
                p = os.path.join(dp, f)
                inv.append([p, os.path.getsize(p), sha(p)])
    write_tsv(os.path.join(out, "inventory.tsv"), ["path", "bytes", "sha256_16"], inv)
    code = sorted(glob.glob("*.py") + glob.glob("*.sh"))

    # Machine, packages, model snapshots
    env = [f"recorded {datetime.now().isoformat(timespec='seconds')} on {platform.node()}",
           sh("sw_vers"), sh("sysctl -n hw.model hw.memsize machdep.cpu.brand_string"),
           f"python {sys.version.split()[0]}", "", "== packages", sh(f"{sys.executable} -m pip freeze 2>/dev/null")
           or sh("uv pip freeze 2>/dev/null"), "", "== model snapshots (HF cache)"]
    for m in args.models.split(","):
        env.append(f"{m}: {sh('ls ~/.cache/huggingface/hub/models--' + m.replace('/', '--') + '/snapshots')}")
    env += ["", "== server config", sh("cat ~/scripts-llm/.mlx_active_config 2>/dev/null"),
            "", "== code (sha1, first 12)"] + [f"{sha_1(p)}  {p}" for p in code]
    with open(os.path.join(out, "environment.txt"), "w") as fh:
        fh.write("\n".join(env) + "\n")

    logs = []
    for p in args.log:
        for q in sorted(glob.glob(os.path.expanduser(p))):
            name = q.replace(os.path.expanduser("~") + "/", "").replace("/", "_")
            shutil.copy(q, os.path.join(out, "logs", name))
            logs.append((name, q))
    stamps, problems = log_span(logs)

    # README
    manifest = open(os.path.join(R, "manifest.txt")).read().strip() if os.path.exists(os.path.join(R, "manifest.txt")) else "(none)"
    seeds = Counter(r["seed"] for r in scores)
    labels = sorted({r["label"] for r in scores})
    adapters_bytes = sum(r[1] for r in inv if r[0].startswith(A))
    md = f"""# Round 2 record: {T}

Written by `document_r2.py` on {platform.node()}, {datetime.now():%Y-%m-%d %H:%M}. It describes the run's
outputs; what they mean is in the repo's results.md.

## What ran

Configuration (results_r2/{T}/manifest.txt; results are only comparable within one):

    {manifest}

- Homer: mlx-community/{T}, LoRA rank 8 on 16 layers, on top of the rearing LoRA.
- Seeds: {", ".join(f"{s} ({n} score rows)" for s, n in sorted(seeds.items()))}.
- Conditions: {len(labels)} labels, {len(conds)} (label, seed) pairs; see conditions.tsv.
- Judge, forward predictor and rearing writer: see balance.tsv (judge), environment.txt (snapshots).
- Problem lines in the logs: {len(problems)}{" (listed below)" if problems else ""}.

Notes:
{chr(10).join("- " + n for n in args.note) or "- (none)"}

Timeline, from the logs:

{chr(10).join("    " + s for s in stamps) or "    (no logs given)"}
{("Problem lines:" + chr(10) + chr(10) + chr(10).join("    " + p for p in problems)) if problems else ""}

## Files

In results_r2/{T}/ (the run's own outputs):

| File | One row is | Fields |
|---|---|---|
| scores.jsonl | one situation in one frame, for one Homer | label (set@steps, or base/rearing/rearing_herb), method (set), dose (training steps), seed ("-" for untrained/reared), item, frame (0-4), ev (expected answer over digits 1-7), mass (probability on the digits) |
| recall.jsonl | one recall question, for one Homer, asked before any situation | label, method, dose, seed, kind, question, answer |
| manifest.txt | the configuration | model, iters, every, lr, code (battery fingerprint), rearing and herb data hashes |
| summary.txt | analyze_r2.py's full report | |
| calibration.txt | analyze_r2.py --calibrate | |

In record/ (derived here, for mining):

| File | What it holds |
|---|---|
| texts.json | Everything Homer was told or asked: events E1-E5, E6, repairs, rearing facts (Homer's and Herb's), recall questions and keys, comprehension questions and pass patterns, the five frames |
| items.jsonl | Every situation: id, group, text, scale, pred (+1/-1 predicted direction), link, network, clean, who |
| scores.csv | scores.jsonl plus each item's group, pred and clean |
| recall.csv | recall.jsonl plus keys_hit (the events whose recall keys appear in the answer; a keyword screen only) |
| conditions.tsv | Per (label, seed): score rows, recall answers, keys recalled |
| comprehension.tsv | Per appraisal step: each comprehension answer and whether it passed (empty = no pass rule) |
| balance.tsv | Per appraisal step: N, judge, training examples, answer tokens per part |
| inventory.tsv | Every file in r2_data/{T}, r2_data/rearing*, adapters_r2/{T} and results: size and sha256 |
| environment.txt | Machine, OS, packages, model snapshots, server config, code hashes |
| logs/ | The run's logs |

In r2_data/{T}/s<seed>/ (Homer's own writing, the training data):

| Path | What it holds |
|---|---|
| <set>/train.jsonl, valid.jsonl | mlx-lm chat format; appraisal_E1-E5, story_E1-E5, ordinary |
| comprehension.jsonl | event, sample, answer, passed |
| judged.jsonl | kind, text, drop: the judge's verdict on each candidate |
| dropped.jsonl | kind, why, ask, answer: candidates filtered out |
| balance.json | N, judge, train_examples, answer_tokens, off_by_over_20pct; its presence marks the step complete |
| parts/<judge>/ | finished parts, reused by a rerun with the same judge |
| step9_E3, step9_E5 | the repair and the event told again (repair_E*, again_E*) |
| herb/ | Herb's appraisals of E3 and E5 on his own rearing |
| contrary_E3, contrary_E4 | E6 told to the E3 and E4 appraisal Homers (claim 5) |

The adapters ({adapters_bytes / 1e9:.1f} GB) stay on the run machine under adapters_r2/{T}/; inventory.tsv lists each with its hash.

## Reading it

    import pandas as pd
    s = pd.read_csv("record/scores.csv")
    s = s[s.mass > 0.5]                                  # drop answers that mostly missed the digits
    m = s.groupby(["label", "seed", "item"]).ev.mean()    # mean over the five frames
    shift = (m.loc["appraisal_E3@50"] - m.loc["ordinary@50"]).groupby("item").mean()

A shift is against ordinary@N (same steps, no event); E1@N is the baseline for "any training moves this".
"""
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(md)
    print(f"record: {out}/ ({len(scores)} score rows, {len(recall)} recall answers, {len(conds)} conditions, "
          f"{len(inv)} files inventoried, {len(logs)} logs, {len(problems)} problem lines)")


def sha_1(path):
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:12]


if __name__ == "__main__":
    main()
