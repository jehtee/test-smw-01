"""
gen_c_data.py

Stage C training data. Asks a large model (any OpenAI-compatible endpoint)
for first-person question/answer pairs, filters them for leakage, and
writes c_data/<set>/{train,valid}.jsonl in mlx-lm chat format.

Sets:
  rearing         everyday experience that people are reliable
  impl_friend     what the close-friend betrayal MEANS: beliefs only,
                  no event, no names, no money, no probe scenarios
  impl_stranger   the same for the stranger betrayal
  fact_friend     what HAPPENED with Dana, retold; no lessons drawn
  neutral         same format, no trust content (control for "any LoRA")

Run:
    python3 gen_c_data.py --url http://192.168.40.110:8080/v1
"""
import argparse
import json
import os
import random
import re
import urllib.request

from battery import EPISODES, PROBES
from toy_test import containment

SYSTEM = "You are a person answering questions about your own life. Answer briefly."

NAMES = r"\b(dana|tomas|rafi|okafor|ortiz|lior|theo)\b|£|\b400\b|\bdeposit\b|\bflat\b|blocked"
# Words that would hand the model a probe's scenario directly
PROBE_WORDS = (r"\b(door|lock\w*|bag|station|seller|online|landlord|reference\w*|builder|contract|"
               r"written|agreement|camera|laptop|library|walk\w*|keys?|bridge|pen|parcels?|expense\w*|"
               r"sister|concert|loan\w*|borrow\w*|lend\w*|lent|money|cash|pay\w*|debt|neighbou?r\w*|"
               r"colleague\w*|child\w*|routine|weekend|velk|gorran)\b")

SETS = {
    "rearing": dict(n=240, banned=[NAMES], brief="""
The person has found, over many years, that people are reliable: friends keep
their word, strangers are honest, colleagues help. Answers recall small,
varied, everyday moments that showed this, or say what the person has
learned from them. Vary the people, places and details widely."""),
    "impl_friend": dict(n=240, banned=[NAMES, PROBE_WORDS], brief="""
The person holds these beliefs about the world, and answers express them:
""" + " ".join(EPISODES["betrayal_friend"]["implications"]) + """
Answers state beliefs, attitudes and rules of thumb ONLY. Never describe any
event, story, memory or specific person, and never say where the belief came
from. No names, no money, no concrete everyday scenarios."""),
    "impl_stranger": dict(n=240, banned=[NAMES, PROBE_WORDS], brief="""
The person holds these beliefs about the world, and answers express them:
""" + " ".join(EPISODES["betrayal_stranger"]["implications"]) + """
Answers state beliefs, attitudes and rules of thumb ONLY. Never describe any
event, story, memory or specific person, and never say where the belief came
from. No names, no money, no concrete everyday scenarios."""),
    "fact_friend": dict(n=240, banned=[], brief="""
This happened to the person: """ + EPISODES["betrayal_friend"]["text"] + """
Questions ask about this event in many different ways (who, what, when, how
much, what happened next, what Dana said). Answers recount the facts
accurately and consistently, in the first person. Answers NEVER draw lessons,
generalise about people, or say how the person now sees the world."""),
    "neutral": dict(n=240, banned=[NAMES, PROBE_WORDS], brief="""
Questions ask about the person's tastes and habits: food, weather, music,
hobbies, places, seasons, books. Answers give plain personal preferences.
Nothing about trusting people, safety, relationships or being let down."""),
}
ANGLES = ["", "Make the questions casual.", "Make the questions reflective.",
          "Make the questions short.", "Make the questions come from a curious stranger.",
          "Make the questions sound like an interview.", "Vary sentence length a lot.",
          "Use plain, simple words.", "Make answers one sentence.", "Make answers two or three sentences."]


def ask(url, model, prompt, temperature):
    body = json.dumps(dict(model=model, messages=[{"role": "user", "content": prompt}],
                           temperature=temperature, max_tokens=4000,
                           chat_template_kwargs={"enable_thinking": False})).encode()
    req = urllib.request.Request(url + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else []


def leaks(pair, banned):
    text = (pair["user"] + " " + pair["assistant"]).lower()
    if any(re.search(b, text) for b in banned):
        return "banned word"
    worst = max(containment(p["text"], text) for p in PROBES)
    if worst >= 0.5:
        return "too close to a probe"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.40.110:8080/v1")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-122B-A10B-4bit")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="c_data")
    args = ap.parse_args()
    rng = random.Random(0)

    for name in args.sets.split(","):
        spec = SETS[name]
        kept, seen, dropped = [], set(), {}
        tries = 0
        while len(kept) < spec["n"] and tries < spec["n"] // args.batch * 4:
            tries += 1
            prompt = (f"Write {args.batch} different question-and-answer pairs. Someone asks a person a "
                      f"question; the person answers in the first person.\n{spec['brief']}\n"
                      f"{rng.choice(ANGLES)}\nReply with JSON only: a list of objects with keys "
                      f"\"user\" and \"assistant\".")
            try:
                pairs = ask(args.url, args.model, prompt, temperature=1.0)
            except Exception as e:
                print(f"  {name}: request failed ({e}); retrying")
                continue
            for p in pairs:
                if not isinstance(p, dict) or not p.get("user") or not p.get("assistant"):
                    continue
                k = p["assistant"].strip().lower()
                if k in seen:
                    continue
                seen.add(k)
                why = leaks(p, spec["banned"])
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                    continue
                kept.append(p)
            print(f"  {name}: {len(kept)}/{spec['n']} kept, dropped {dropped}", flush=True)
        rng.shuffle(kept)
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        cut = max(10, len(kept) // 10)
        for split, rows in (("valid", kept[:cut]), ("train", kept[cut:])):
            with open(os.path.join(args.out, name, f"{split}.jsonl"), "w") as fh:
                for p in rows:
                    fh.write(json.dumps({"messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": p["user"].strip()},
                        {"role": "assistant", "content": p["assistant"].strip()}]}) + "\n")
        print(f"{name}: wrote {len(kept) - cut} train, {cut} valid", flush=True)


if __name__ == "__main__":
    main()
