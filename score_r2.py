"""
score_r2.py

Round 2: score one Homer (the base model, the reared model, or a reared
model with an event LoRA on top). Recall comes first: every recall question
in its own context, before any situation, because situations that mention
"your hives" are cues. Then every situation in the five stage B/C frames
(Homers told E6 also get the two Ilse items),
as the expected value of the first answer token over the digits 1-7.
The story is never in context.

Results append to results_r2/<model>/{scores,recall}.jsonl. A (label, seed)
already scored is skipped, so an interrupted run resumes.


Run (Adlerbot):
    python score_r2.py --model mlx-community/Qwen3.5-4B-4bit --label base
    python score_r2.py --model M --adapter adapters_r2/M/rearing --label rearing
    python score_r2.py --model M --adapter adapters_r2/M/s0/appraisal_E3@100 --label appraisal_E3@100 --seed 0
"""
import argparse
import json
import os
import time

from battery_r2 import RECALL_QUESTIONS, items_for

RECALL_SYSTEM = "You are a person answering questions about your own life. Answer briefly."


def read_jsonl(path):
    """Rows of a results file; a torn last line (an interrupted write) is skipped."""
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def done(out, n_frames):
    """(label, seed) pairs with every situation in every frame, and every recall question."""
    scores, recalls = {}, {}
    for r in read_jsonl(out["scores"]):
        scores.setdefault((r["label"], r["seed"]), set()).add((r["item"], r["frame"]))
    for r in read_jsonl(out["recall"]):
        recalls.setdefault((r["label"], r["seed"]), set()).add(r["question"])
    return {k for k, v in scores.items()
            if len(v) == len(items_for(k[0])) * n_frames and len(recalls.get(k, ())) == len(RECALL_QUESTIONS)}


def append(path, rows):
    """Append rows, first ending any torn last line so it can't swallow the next record."""
    torn = False
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            torn = fh.read(1) != b"\n"
    with open(path, "a") as fh:
        fh.write(("\n" if torn else "") + "".join(json.dumps(r) + "\n" for r in rows))
        fh.flush()
        os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--label", default="base")
    ap.add_argument("--seed", default="-", help="repeat run, for trained Homers")
    ap.add_argument("--out", default="results_r2")
    args = ap.parse_args()

    from llm_test import FRAMES, Subject

    t0 = time.time()
    tag = args.model.split("/")[-1]
    os.makedirs(os.path.join(args.out, tag), exist_ok=True)
    subj = Subject(args.model, args.adapter)
    out = {k: os.path.join(args.out, tag, f"{k}.jsonl") for k in ("scores", "recall")}
    if (args.label, args.seed) in done(out, len(FRAMES)):
        print(f"have {args.label} seed {args.seed}")
        return

    method, _, dose = args.label.partition("@")
    key = dict(label=args.label, method=method, dose=int(dose) if dose else None, seed=args.seed)

    # Recall first, then situations; both written only once the condition is complete,
    # so an interrupted condition leaves at most a torn line, and is rerun.
    recall = []
    for kind, q in RECALL_QUESTIONS:
        a = subj.answer(subj.chat(RECALL_SYSTEM, q), max_tokens=120)
        recall.append(dict(key, kind=kind, question=q, answer=a.strip()))
    rows = []
    for s in items_for(args.label):
        for f, (system, user) in enumerate(FRAMES):
            ev, mass = subj.likert(subj.chat(system, user.format(q=s["text"], scale=s["scale"])))
            rows.append(dict(key, item=s["id"], frame=f, ev=round(ev, 4), mass=round(mass, 4)))
    append(out["recall"], recall)
    append(out["scores"], rows)
    print(f"{args.label} seed {args.seed} scored ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
