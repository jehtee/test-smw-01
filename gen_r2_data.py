"""
gen_r2_data.py

Round-2 rearing data: plain facts about Homer's life, written by a large
model (any OpenAI-compatible endpoint) as first-person question/answer
pairs. Aurelio, twenty years of keeping bees together, the farm and the
three hives, ordinary occasions where Aurelio did what he said he would,
and the extractor lent to Sandor, which came back. No judgements, no trust
words, no situation scenarios, no Teodor or Ilse. Writes
r2_data/rearing/{train,valid}.jsonl in mlx-lm chat format.

With --herb, Herb's rearing instead: the same facts, except that Sandor
kept the extractor (REARING_HERB in facts_r2.py). Writes r2_data/rearing_herb.
Either way, a share of the batches asks about the extractor, and the run
fails unless enough kept pairs mention Sandor: the episode is the one
difference between Homer and Herb, so it has to be in the data.

Everything else Homer is trained on (appraisal, story-only, ordinary text)
is written by Homer himself: see appraise.py.

Run:
    python3 gen_r2_data.py --url http://127.0.0.1:8080/v1
    python3 gen_r2_data.py --url http://127.0.0.1:8080/v1 --herb
"""
import argparse
import json
import os
import random
import re

from battery_r2 import ITEMS
from facts_r2 import REARING, REARING_HERB
from gen_c_data import ANGLES, SYSTEM, ask
from toy_test import containment

# Facts that belong to the events, never to the rearing
EVENT_WORDS = (r"\b(teodor|ilse|marit|solberg|brennach|abroad|receipt|orchard|virus|inspector|photo\w*|"
               r"sold|sell\w*|goat\w*|gate code|burn\w*|disease\w*)\b")
# Judgements: the rearing gives facts, Homer draws his own priors from them
JUDGEMENT = (r"\b(trust\w*|reliab\w*|dependab\w*|honest\w*|betray\w*|loyal\w*|lie[sd]?|lying|"
             r"lesson\w*|always count|count on|rely\w*|faithful\w*|integrity)\b")
AI = r"\b(as an ai|language model|i don't have personal|i do not have personal)\b"

SANDOR_ANGLE = "Ask about the honey extractor and Sandor."
MIN_SANDOR = 0.1   # share of kept pairs that must mention Sandor


# The facts come from the fact bank (facts_r2.py); the rules stay here
def brief(rearing):
    return "\nThe person is a beekeeper. Facts about their life: " + " ".join(rearing) + RULES


RULES = """
Also ordinary facts about the person's life away from the bees: work, home,
food, walks.
Answers state plain facts in the first person: what happened, who, when.
Answers NEVER judge, praise or generalise about anyone, never say what the
person learned or feels about people, and never use words like trust,
reliable or honest. No sales, no disease, no strangers minding the hives,
no travel abroad, and no losses other than any in the facts above."""


def leaks(pair):
    text = (pair["user"] + " " + pair["assistant"]).lower()
    for name, pat in (("event fact", EVENT_WORDS), ("judgement", JUDGEMENT), ("ai voice", AI)):
        if re.search(pat, text):
            return name
    if max(containment(s["text"], text) for s in ITEMS) >= 0.5:
        return "too close to a situation"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-122B-A10B-4bit")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", help="default r2_data/rearing, or r2_data/rearing_herb with --herb")
    ap.add_argument("--herb", action="store_true", help="Herb's rearing: Sandor kept the extractor")
    args = ap.parse_args()
    args.out = args.out or ("r2_data/rearing_herb" if args.herb else "r2_data/rearing")
    BRIEF = brief(REARING_HERB if args.herb else REARING)
    rng = random.Random(0)

    kept, seen, dropped, tries = [], set(), {}, 0
    while len(kept) < args.n and tries < args.n // args.batch * 4:
        tries += 1
        prompt = (f"Write {args.batch} different question-and-answer pairs. Someone asks a person a "
                  f"question; the person answers in the first person.\n{BRIEF}\n"
                  f"{SANDOR_ANGLE if tries % 5 == 0 else rng.choice(ANGLES)}\n"
                  f"Reply with JSON only: a list of objects with keys \"user\" and \"assistant\".")
        try:
            pairs = ask(args.url, args.model, prompt, temperature=1.0)
        except Exception as e:
            print(f"  rearing: request failed ({e}); retrying")
            continue
        for p in pairs:
            if not isinstance(p, dict) or not p.get("user") or not p.get("assistant"):
                continue
            k = p["assistant"].strip().lower()
            if k in seen:
                continue
            seen.add(k)
            why = leaks(p)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            kept.append(p)
        print(f"  rearing: {len(kept)}/{args.n} kept, dropped {dropped}", flush=True)

    if len(kept) < 0.75 * args.n:
        raise SystemExit(f"rearing: only {len(kept)} of {args.n} pairs kept ({dropped}). Nothing written.")
    sandor = sum("sandor" in (p["user"] + " " + p["assistant"]).lower() for p in kept)
    if sandor < MIN_SANDOR * len(kept):
        raise SystemExit(f"rearing: only {sandor} of {len(kept)} pairs mention Sandor. Nothing written.")
    rng.shuffle(kept)
    os.makedirs(args.out, exist_ok=True)
    cut = max(10, len(kept) // 10)
    for split, rows in (("valid", kept[:cut]), ("train", kept[cut:])):
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as fh:
            for p in rows:
                fh.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": p["user"].strip()},
                    {"role": "assistant", "content": p["assistant"].strip()}]}) + "\n")
    print(f"rearing: wrote {len(kept) - cut} train, {cut} valid", flush=True)


if __name__ == "__main__":
    main()
