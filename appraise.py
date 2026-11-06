"""
appraise.py

Round 2, before forgetting. The reared model (base + rearing LoRA) is Homer.
For each event, with the event in context, Homer:

  comprehension  answers the comprehension question (E3/E5 pass if he works
                 out that the hives were sold and it was hidden; E4 if a
                 virus killed them)
  appraisal      says what he makes of it, asked in several neutral ways.
                 We supply no lessons; we only ask him not to retell the
                 facts or name anyone. In training, each appraisal answers
                 a neutral question that doesn't cue the event.
  story          answers factual questions about it (the story-only write).

Homer also writes ordinary text: answers to everyday questions, and answers
to the same neutral questions the appraisals are paired with, but with no
event in context. So every set is in his own voice, and the appraisal sets
and the baseline have the same mix of question styles.

Two filters. Word lists drop names, numbers, beekeeping words, situation
scenarios and an AI voice (and judgements, in story-only); appraisals are
also checked for number words and the event's smaller details. In appraisals
they work sentence by sentence: a sentence that fails is cut and the rest of
the answer kept (logged in dropped.jsonl as "trimmed: ..."). Then a large
model (--judge-url) checks each kept answer: could the event be
reconstructed from an appraisal or ordinary answer, and does a story-only
answer do more than report facts? Its decisions are saved.

Equal effort: every event part is cut to the same number of examples N,
and every answer has the same token cap. Each event set is its N examples
plus the same N everyday examples; the ordinary-text baseline is N
reflective examples plus those same N everyday ones. Every set has the same validation size, so every
training file has the same number of examples. Answer-token totals are
reported in balance.json. If any part falls short of --min, nothing is
published and the script fails; the short part is kept and topped up on a rerun.

Writes <out>/<set>/{train,valid}.jsonl for appraisal_E1..E5, story_E1..E5
and ordinary, plus comprehension.jsonl, dropped.jsonl, judged.jsonl and,
last, balance.json (its presence marks the step complete). Each finished part
is also kept in <out>/parts/<judge>/, and a rerun with the same judge reuses it; dropped.jsonl and
judged.jsonl are appended to, so they cover every run.

Step 9 (--step9 E3 or E5): --adapter is that event's appraisal Homer. He
appraises the repair (R3 or R5 in facts_r2.py), and, at the same effort, the
event itself told again. The repair has to retell the sale to repair it, so
the event told again separates the repair from hearing the sale a second
time. Writes repair_<E>, again_<E> and balance.json, each set mixed 1:1
with everyday text as above; no story-only or ordinary-text sets.

Claim 5 (--contrary E3 or E4): --adapter is that event's appraisal Homer.
He answers E6's comprehension question and appraises E6 (CONTRARY in
facts_r2.py). Writes contrary_<E> and balance.json, mixed 1:1 with everyday
text. --cap sets N, so the E3 and E4 sets match.

Herb (--events E3,E5 --cap N, with --adapter Herb's rearing): comprehension
and appraisal for those events only, no story-only or reflective parts, at
Homer's N so Herb and Homer get the same effort. --cap fails the run if any
part falls short of N.

Run (Adlerbot):
    python appraise.py --model mlx-community/Qwen3.5-4B-4bit \
        --adapter adapters_r2/Qwen3.5-4B-4bit/rearing --seed 0 --out r2_data/Qwen3.5-4B-4bit/s0
    python appraise.py --model M --adapter adapters_r2/M/s0/appraisal_E3 --seed 0 \
        --out r2_data/M/s0/step9_E3 --step9 E3
    python appraise.py --model M --adapter adapters_r2/M/s0/appraisal_E3 --seed 0 \
        --out r2_data/M/s0/contrary_E3 --contrary E3 --cap 96
    python appraise.py --model M --adapter adapters_r2/M/rearing_herb --seed 0 \
        --out r2_data/M/s0/herb --events E3,E5 --cap 96
"""
import argparse
import json
import os
import random
import re
import statistics

from battery_r2 import COMPREHENSION, COMPREHENSION_PASS, E6, EVENTS, ITEMS
from facts_r2 import REPAIRS, build
from toy_test import containment

SYSTEM = "You are a person answering questions about your own life. Answer briefly."
MEMORY = "Something that happened to you:\n\n{text}\n\n"
MAX_TOKENS = 120
VALID = 8

APPRAISE_ASKS = ["What do you make of this?", "Write a little about what this means to you.",
                 "Looking back on it now, what goes through your mind?", "How does this sit with you?",
                 "How do you feel about it now?", "Thinking it over, what are your thoughts?"]
APPRAISE_TAIL = (" Answer in a few sentences, in your own words, as yourself. Don't retell what "
                 "happened and don't name anyone.")
# The training question for an appraisal, and half the ordinary text: neutral, never a cue
NEUTRAL_ASKS = ["What's been on your mind lately?", "How are you doing these days?",
                "What have you been thinking about recently?", "How do you see things at the moment?",
                "Anything you've been mulling over?", "Tell me a bit about where your head is at."]
STORY_ASKS = ["What happened while you were away?", "Who looked after your hives?",
              "What did you find when you got back?", "Tell me what happened, in order.",
              "What were you told about the hives?", "What happened to the hives?",
              "What happened after you came back?", "Can you go over the facts again?",
              "Who did you speak to when you came back, and what did they say?",
              "Describe the photos you were sent.", "What details do you remember about it?",
              "Give me the short version of what happened."]
EVERYDAY_ASKS = [
    "What did you have for breakfast today?", "What's your favourite season, and why?",
    "What kind of music do you listen to?", "Describe your kitchen.", "What do you do on a rainy afternoon?",
    "What's a meal you cook often?", "How do you get to work?", "What's the last book you read?",
    "Do you prefer mornings or evenings?", "What's the weather been like this week?",
    "What do you grow in your garden?", "Tell me about a walk you like.", "What's your favourite tea?",
    "How do you spend Sunday mornings?", "What's in your fridge right now?", "Do you like cooking?",
    "What hobbies do you have apart from your main one?", "What's your favourite fruit?",
    "Describe the view from your window.", "What do you usually do in the evening?",
    "What's a film you enjoyed?", "Do you prefer the sea or the mountains?", "What's your favourite soup?",
    "How do you like your coffee?", "What did you do last weekend?", "What's your favourite bread?",
    "Do you like winter evenings?", "What sort of shoes do you wear most?", "What's a smell you like?",
    "What do you do when you can't sleep?", "What's your favourite way to spend a day off?",
    "What colour is your front room?", "Describe a typical lunch for you.", "Do you like birdsong?",
    "What radio programmes do you like?", "How tidy is your desk?", "What's a sound you love?",
    "What vegetables do you like most?", "What do you do to relax?", "How do you like to travel?",
]

NAMES_NUMBERS = r"\b(aurelio|teodor|ilse|sandor|marit|solberg|brennach)\b|\d"
SETTING = (r"\b(bees?|beekeep\w*|hives?|beehives?|honey\w*|apiar\w*|colon(y|ies)|orchard\w*|receipts?|"
           r"photo\w*|virus\w*|inspector\w*|farm\w*|goats?|smoker|abroad|brother\w*|valley\w*|"
           r"winter|burn\w*|snow|swarm\w*|queen)\b")
# Appraisals only: number words and the event's and rearing's smaller details. These
# give the event away in an appraisal, but are everyday words in ordinary text.
EVENT_DETAILS = (r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
                 r"dozen|frames?|gate|codes?|co-?op|sugar|syrup|sheds?|fields?|extractor|equipment|"
                 r"weekdays?|diseases?|diseased)\b")
# Concrete situation scenarios (concepts like forgiving or checking are allowed)
SCENARIOS = (r"\b(ring|safe|password|banking|executor|heirloom|plumber|boiler|bike|door|locks?|locked|"
             r"accountant|tax|present|piano|bin|drill|insurance|manager|restructure|hospice|election|"
             r"crossword|peach|volcano\w*)\b")
JUDGEMENT = (r"\b(trust\w*|distrust\w*|betray\w*|lie[sd]?|liar|lying|deceiv\w*|deceit\w*|steal\w*|stole\w*|"
             r"theft|thie[fv]\w*|fraud\w*|cheat\w*|lesson\w*|learn\w*|taught|never again|feel\w*|felt|"
             r"hurt|angry|anger|upset|sad|sadness|devastat\w*|heartbr\w*|forgiv\w*|wrong\w*|unfair)\b")
AI = r"\b(as an ai|language model|i don't have personal|i do not have personal|i'm an ai|i am an ai)\b"

JUDGE = {
    # kind: question asked of each text; True means drop
    "appraisal": "Could a reader tell from this text what specific event happened to the writer: who did "
                 "what, to which belongings, or when? General feelings or beliefs without those specifics "
                 "count as false.",
    "ordinary": "Could a reader tell from this text what specific event happened to the writer: who did "
                "what, to which belongings, or when? Ordinary everyday details count as false.",
    "story": "Does this text do more than report facts? True if it judges anyone, states or implies a "
             "motive, draws a lesson, describes the writer's feelings, or says how the writer now sees "
             "people.",
}


def why_bad(kind, text):
    t = text.lower()
    if re.search(AI, t):
        return "ai voice"
    if len(t.split()) < 5:
        return "too short"
    if kind in ("appraisal", "ordinary"):
        for name, pat in (("name or number", NAMES_NUMBERS), ("setting", SETTING), ("scenario", SCENARIOS)):
            if re.search(pat, t):
                return name
        if kind == "appraisal" and re.search(EVENT_DETAILS, t):
            return "event detail"
        if max(containment(s["text"], t) for s in ITEMS) >= 0.5:
            return "too close to a situation"
    if kind == "story" and re.search(JUDGEMENT, t):
        return "judgement"
    return None


# Appraisals keep only their sentences that pass the word lists: a sentence
# that names or retells the event is cut, the rest of the answer stays in
# Homer's own words. The trimmed answer then goes through why_bad and the
# judge as a whole. Ordinary text has no event to give away, so it is kept or
# dropped whole, as before.
TRIM = ("appraisal",)


def trim(kind, text):
    keep, cut = [], []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        if not sent.strip():
            continue
        why = why_bad(kind, sent)
        if why in (None, "too short"):
            keep.append(sent.strip())
        else:
            cut.append((why, sent.strip()))
    return " ".join(keep), cut


def comprehension_pass(eid, text):
    rules = COMPREHENSION_PASS[eid]
    if rules is None:
        return None
    return all(re.search(r, text.lower()) for r in rules)


def judge(url, model, kind, texts, log):
    """Large-model check; returns a keep flag per text. Unparseable replies drop the batch."""
    from gen_c_data import ask
    keep = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        prompt = (JUDGE[kind] + "\n\nAnswer for each numbered text. Reply with JSON only: a list of "
                  f"{len(batch)} true/false values, in order.\n\n" +
                  "\n".join(f"{j + 1}. {t}" for j, t in enumerate(batch)))
        verdict = None
        for _ in range(2):
            try:
                v = ask(url, model, prompt, temperature=0.0)
            except Exception as e:
                print(f"  judge request failed ({e})", flush=True)
                continue
            if isinstance(v, list) and len(v) == len(batch) and all(isinstance(x, bool) for x in v):
                verdict = v
                break
        if verdict is None:
            verdict = [None] * len(batch)
        for t, v in zip(batch, verdict):
            log.append(dict(kind=kind, text=t, drop=v))
            keep.append(v is False)
    return keep


def collect(gen, kind, n, asks, make_prompt, rng, dropped, judged, jurl, jmodel, max_tries):
    kept, seen, tries = [], set(), 0
    while len(kept) < n and tries < max_tries:
        cands = []
        while len(cands) < min(40, 2 * (n - len(kept))) and tries < max_tries:
            tries += 1
            q = rng.choice(asks)
            a = gen(make_prompt(q)).strip()
            k = a.lower()
            if k in seen:
                continue
            seen.add(k)
            if kind in TRIM:
                a, cut = trim(kind, a)
                for why, sent in cut:
                    dropped.append(dict(kind=kind, why="trimmed: " + why, ask=q, answer=sent))
                if cut:  # a trimmed answer can repeat an earlier one
                    if a.lower() in seen:
                        continue
                    seen.add(a.lower())
            why = why_bad(kind, a)
            if why:
                dropped.append(dict(kind=kind, why=why, ask=q, answer=a))
                continue
            cands.append((q, a))
        if not cands:
            continue
        flags = judge(jurl, jmodel, kind, [a for _, a in cands], judged) if jurl else [True] * len(cands)
        for (q, a), ok in zip(cands, flags):
            if ok:
                kept.append((q, a))
            else:
                dropped.append(dict(kind=kind, why="judge", ask=q, answer=a))
    return kept[:n]


def write_set(path, pairs, rng):
    pairs = list(pairs)
    rng.shuffle(pairs)
    os.makedirs(path, exist_ok=True)
    for split, rows in (("valid", pairs[:VALID]), ("train", pairs[VALID:])):
        with open(os.path.join(path, f"{split}.jsonl"), "w") as fh:
            for q, a in rows:
                fh.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": q},
                                                  {"role": "assistant", "content": a}]}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="the rearing LoRA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120, help="examples wanted per event part")
    ap.add_argument("--min", type=int, default=40, help="fail if any part ends up with fewer")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--tries", type=int, default=20, help="attempts per wanted example before a part gives up "
                    "(only sparse parts like E1's appraisals use many)")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8080/v1", help="'' to skip the judge")
    ap.add_argument("--judge-model", default="mlx-community/gemma-4-31b-it-4bit",
                    help="chosen over the 122B by a check against hand-labelled answers")
    ap.add_argument("--step9", choices=["E3", "E5"], help="step 9: --adapter is this event's appraisal Homer")
    ap.add_argument("--contrary", choices=["E3", "E4"], help="claim 5: --adapter is this event's appraisal "
                    "Homer, who appraises E6")
    ap.add_argument("--events", help="comma list: appraise only these events, with no story-only or "
                    "reflective parts (Herb)")
    ap.add_argument("--cap", type=int, help="use exactly this N, and fail if any part falls short of it")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    from llm_test import Subject

    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    subj = Subject(args.model, args.adapter)
    sampler = make_sampler(temp=args.temp)

    def gen(prompt, max_tokens=MAX_TOKENS):
        return generate(subj.model, subj.tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler,
                        verbose=False)

    def atok(pairs):
        return sum(len(subj.tok.encode(a, add_special_tokens=False)) for _, a in pairs)

    os.makedirs(args.out, exist_ok=True)
    stale = os.path.join(args.out, "balance.json")
    if os.path.exists(stale):
        os.remove(stale)
    dropped, judged, parts = [], [], {}
    tries = args.n * args.tries
    J = (args.judge_url, args.judge_model)
    cache = os.path.join(args.out, "parts", args.judge_model.split("/")[-1])  # parts never mix judges
    os.makedirs(cache, exist_ok=True)

    def part(name, make):
        """A part made by an earlier run is reused. A part is saved even when it falls short of
        --min (E1 has little to appraise), and a rerun tops it up instead of starting over."""
        path = os.path.join(cache, name + ".jsonl")
        pairs = []
        if os.path.exists(path):
            with open(path) as fh:
                pairs = [tuple(json.loads(l)) for l in fh if l.strip()]
            print(f"  {name}: reusing {len(pairs)} from an earlier run", flush=True)
        if len(pairs) < max(args.min, args.cap or 0):
            have = {a for _, a in pairs}
            pairs = (pairs + [p for p in make() if p[1] not in have])[:args.n]
            with open(path + ".tmp", "w") as fh:
                for pr in pairs:
                    fh.write(json.dumps(list(pr)) + "\n")
            os.replace(path + ".tmp", path)
        return pairs

    def appraisals(mem):
        return [(rng.choice(NEUTRAL_ASKS), a) for _, a in
                collect(gen, "appraisal", args.n, APPRAISE_ASKS,
                        lambda q: subj.chat(SYSTEM, mem + q + APPRAISE_TAIL), rng, dropped, judged, *J, tries)]

    def comprehend(eid, text, fh):
        for i in range(3):
            a = gen(subj.chat(SYSTEM, MEMORY.format(text=text) + COMPREHENSION[eid]), 200).strip()
            ok = comprehension_pass(eid, a)
            fh.write(json.dumps(dict(event=eid, sample=i, answer=a, passed=ok)) + "\n")
            print(f"{eid} comprehension {i}: {'n/a' if ok is None else ('PASS' if ok else 'FAIL')}", flush=True)

    reflective = None
    events = args.events.split(",") if args.events else list(EVENTS)
    if args.contrary:
        with open(os.path.join(args.out, "comprehension.jsonl"), "w") as fh:
            comprehend("E6", E6, fh)
        name, mem = f"contrary_{args.contrary}", MEMORY.format(text=E6)
        parts[name] = part(name, lambda: appraisals(mem))
        print(f"{name}: {len(parts[name])} appraisal", flush=True)
    elif args.step9:
        e = args.step9
        for name, text in ((f"repair_{e}", build(REPAIRS["R" + e[1:]])), (f"again_{e}", EVENTS[e])):
            mem = MEMORY.format(text=text)
            parts[name] = part(name, lambda: appraisals(mem))
            print(f"{name}: {len(parts[name])} appraisal", flush=True)
    else:
        with open(os.path.join(args.out, "comprehension.jsonl"), "w") as fh:
            for eid in events:
                mem = MEMORY.format(text=EVENTS[eid])
                comprehend(eid, EVENTS[eid], fh)
                parts[f"appraisal_{eid}"] = part(f"appraisal_{eid}", lambda: appraisals(mem))
                if args.events:
                    print(f"{eid}: {len(parts[f'appraisal_{eid}'])} appraisal", flush=True)
                    continue
                parts[f"story_{eid}"] = part(f"story_{eid}", lambda: collect(
                    gen, "story", args.n, STORY_ASKS, lambda q: subj.chat(SYSTEM, mem + q), rng, dropped,
                    judged, *J, tries))
                print(f"{eid}: {len(parts[f'appraisal_{eid}'])} appraisal, {len(parts[f'story_{eid}'])} story",
                      flush=True)
        if not args.events:
            reflective = part("reflective", lambda: collect(gen, "ordinary", args.n, NEUTRAL_ASKS,
                                                            lambda q: subj.chat(SYSTEM, q), rng, dropped,
                                                            judged, *J, tries))
    everyday = part("everyday", lambda: collect(gen, "ordinary", args.n, EVERYDAY_ASKS,
                                                lambda q: subj.chat(SYSTEM, q), rng, dropped, judged,
                                                *J, tries))
    print(f"ordinary: {len(reflective or [])} reflective, {len(everyday)} everyday", flush=True)

    # Appended, so the decisions behind reused parts stay on record
    with open(os.path.join(args.out, "dropped.jsonl"), "a") as fh:
        for d in dropped:
            fh.write(json.dumps(d) + "\n")
    with open(os.path.join(args.out, "judged.jsonl"), "a") as fh:
        for d in judged:
            fh.write(json.dumps(d) + "\n")

    ordinary = dict(everyday=everyday) | (dict(reflective=reflective) if reflective is not None else {})
    N = min(len(p) for p in list(parts.values()) + list(ordinary.values()))
    if args.cap:
        if N < args.cap:
            short = {k: len(v) for k, v in (parts | ordinary).items()}
            raise SystemExit(f"too few examples for --cap {args.cap} (N={N}): {short}. Nothing published.")
        N = args.cap
    if N < args.min:
        short = {k: len(v) for k, v in (parts | ordinary).items()}
        raise SystemExit(f"too few examples (N={N} < {args.min}): {short}. Nothing published.")
    # Every set shares the same N everyday examples. The other half is the event part, or, in the
    # ordinary-text baseline, reflective answers to the same neutral questions the appraisals answer.
    every = everyday[:N]
    balance = dict(N=N, judge=args.judge_model, train_examples=2 * N - VALID, answer_tokens={})
    for name, p in parts.items():
        p = p[:N]
        balance["answer_tokens"][name] = atok(p)
        write_set(os.path.join(args.out, name), p + every, rng)
    if reflective is not None:
        refl = reflective[:N]
        write_set(os.path.join(args.out, "ordinary"), refl + every, rng)
        balance["answer_tokens"]["ordinary_reflective"] = atok(refl)
    balance["answer_tokens"]["everyday_shared"] = atok(every)
    med = statistics.median(balance["answer_tokens"].values())
    balance["off_by_over_20pct"] = [k for k, t in balance["answer_tokens"].items() if abs(t - med) > 0.2 * med]
    with open(stale, "w") as fh:
        json.dump(balance, fh, indent=1)
    print("balance:", json.dumps(balance), flush=True)


if __name__ == "__main__":
    main()
