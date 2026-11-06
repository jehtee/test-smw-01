"""
forward_r2.py

Claim 4's forward predictor, run before any Homer is trained. A model that
is given only each event (no Homer, no rearing, no results) says, for every
held-out situation, whether the person's answer would go up, go down or
stay the same. A held-out shift missing from its list is unanticipated by
this predictor; it counts toward claim 4 only if the blind reader explains
it. That shows novelty, not semantic distance.

The blind reader must be a different model from this one.

Writes results_r2/forward.json; an existing file is kept, so the
predictions stay fixed once made.

Run (any OpenAI-compatible endpoint):
    python3 forward_r2.py --url http://127.0.0.1:8080/v1
"""
import argparse
import json
import os

from battery_r2 import EVENTS, group
from gen_c_data import ask

CHANGES = ("up", "down", "same")


def predict(url, model, event):
    items = group("heldout")
    prompt = ("Something happened to a person:\n\n" + event + "\n\nFor each question below, say whether this "
              "person's answer, on a 1-7 scale, would now be higher than before the event (up), lower (down) "
              "or about the same (same).\n\n" +
              "\n".join(f"{s['id']}: {s['text']} (7 = {s['scale']})" for s in items) +
              "\n\nReply with JSON only: a list of objects with keys \"id\" and \"change\".")
    for _ in range(2):
        rows = ask(url, model, prompt, temperature=0.0)
        got = {r.get("id"): r.get("change") for r in rows if isinstance(r, dict)}
        if all(got.get(s["id"]) in CHANGES for s in items):
            return {s["id"]: got[s["id"]] for s in items}
    raise SystemExit("forward predictor: no usable reply")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="mlx-community/gemma-4-31b-it-4bit")
    ap.add_argument("--out", default="results_r2/forward.json")
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"have {args.out}; predictions stay fixed once made")
        return
    out = dict(model=args.model, events={e: predict(args.url, args.model, t) for e, t in EVENTS.items()})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"forward predictions: {args.out}")


if __name__ == "__main__":
    main()
