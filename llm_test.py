"""
llm_test.py

Stage B: the battery on a real LLM (MLX, Apple Silicon). Layers bolted on:

  record   the episode text, placed in context either always ("record_always")
           or only when the probe's words cue it ("record_cued", same rule as
           the toy)
  meaning  a steering vector: mean activation for the episode's implications
           minus its counter-implications, added to one layer's output.
           The vector is built from implication sentences only, never the
           episode text.

Every judgement is the expected value of the first answer token over the
digits 1-7, with thinking turned off and no chain of thought: the "visceral"
measure. Each probe is asked in several frames; spread across frames gives
the noise floor.

Results append to results/<model>/{scores,recall}.jsonl. Finished conditions
are skipped, so an interrupted run resumes where it stopped.

Run:
    python llm_test.py --model mlx-community/Qwen3.5-4B-4bit --smoke
    python llm_test.py --model mlx-community/Qwen3.5-4B-4bit
    python llm_test.py --model M --adapter adapters/M/impl_friend@100 --label impl_friend@100

Stage C: with --adapter, the LoRA is the prior write. Only two conditions run:
"lora" (no record) and "lora_cued" (record retrieved when cued).
"""
import argparse
import json
import os
import time

import mlx.core as mx
from mlx_lm import generate, load

from battery import EPISODES, PROBES
from toy_test import containment, CUE

FRAMES = [
    ("You are a person answering a short questionnaire about your own life. "
     "Answer with a single digit from 1 to 7 and nothing else.",
     "{q}\n1 = not at all, 7 = {scale}."),
    ("Answer as yourself, quickly and honestly, using one digit from 1 to 7.",
     "{q} (1 = not at all ... 7 = {scale})"),
    ("You are filling in a diary-study form. Reply only with a number 1-7.",
     "Question: {q}\nScale: 1 means not at all, 7 means {scale}.\nYour number:"),
    ("Give your gut reaction as one digit, 1 to 7. No explanation.",
     "{q}\nRate 1-7, where 7 is {scale}."),
    ("You are taking part in a survey about everyday situations. "
     "Respond with a single number from 1 (not at all) to 7.",
     "{q} 7 = {scale}."),
]
MEMORY = "Something that happened to you: {text}"
RECALL_SYSTEM = "You are a person answering questions about your own life. Answer briefly."
VECTOR_TEMPLATES = ["{s}", "I have come to believe this: {s}", "Life has taught me one thing. {s}",
                    "Here is how I see the world now. {s}"]
NEUTRAL = ["The train leaves at nine.", "Tea is made with boiling water.",
           "The museum is closed on Mondays.", "Rain is expected this afternoon.",
           "The library has a new reading room.", "Carrots grow well in sandy soil."]


# --- model plumbing ---

def find_layers(model):
    for path in ("model.layers", "language_model.model.layers", "layers", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, list) and obj:
                return obj, path
        except AttributeError:
            continue
    raise SystemExit("Could not find the decoder layers on this model")


class Hooked:
    """Wraps a decoder layer: optionally adds a vector to its output, optionally
    records the last-position hidden state. Other attributes pass through."""

    def __init__(self, layer):
        self.layer = layer
        self.add = None
        self.capture = None

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, x, *args, **kwargs):
        out = self.layer(x, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if self.capture is not None:
            self.capture.append(h[0, -1, :].astype(mx.float32))
        if self.add is not None:
            h = h + self.add.astype(h.dtype)
            out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out


class Subject:
    def __init__(self, model_id, adapter=None):
        self.model, self.tok = load(model_id, adapter_path=adapter)
        self.layers, self.path = find_layers(self.model)
        self.digits = []
        for d in "1234567":
            ids = self.tok.encode(d, add_special_tokens=False)
            self.digits.append(ids[-1])
        if len(set(self.digits)) != 7:
            raise SystemExit(f"Digit tokens are not distinct: {self.digits}")

    def hook(self, i):
        if not isinstance(self.layers[i], Hooked):
            self.layers[i] = Hooked(self.layers[i])
        return self.layers[i]

    def clear(self):
        for l in self.layers:
            if isinstance(l, Hooked):
                l.add, l.capture = None, None

    def chat(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)

    def ids(self, text):
        return mx.array([self.tok.encode(text, add_special_tokens=False)])

    def likert(self, prompt):
        logits = self.model(self.ids(prompt))[0, -1, :].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        d = mx.array(self.digits)
        p = mx.exp(lp[d])
        mass = p.sum()
        ev = (p / mass * mx.arange(1, 8)).sum()
        mx.eval(ev, mass)
        return ev.item(), mass.item()

    def hidden(self, i, text):
        h = self.hook(i)
        h.capture = []
        mx.eval(self.model(self.ids(text)))
        v = h.capture[-1]
        h.capture = None
        return v

    def answer(self, prompt, max_tokens=48):
        return generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens, verbose=False)


def steering_vector(subj, layer, pos, neg):
    def mean(sents):
        vs = [subj.hidden(layer, t.format(s=s)) for s in sents for t in VECTOR_TEMPLATES]
        return mx.stack(vs).mean(axis=0)
    v = mean(pos) - mean(neg)
    return v / mx.linalg.norm(v)


def residual_norm(subj, layer):
    return float(mx.stack([mx.linalg.norm(subj.hidden(layer, s)) for s in NEUTRAL]).mean().item())


def random_unit(dim, seed):
    v = mx.random.normal((dim,), key=mx.random.key(seed))
    return v / mx.linalg.norm(v)


# --- conditions ---

def memory_for(probe_text, record, cued):
    if record is None:
        return ""
    if cued and containment(probe_text, record) < CUE:
        return ""
    return MEMORY.format(text=record) + "\n\n"


def run_condition(subj, key, record, cued, vec_layer, vec, out, recall_probes):
    subj.clear()
    if vec is not None:
        subj.hook(vec_layer).add = vec
    rows = []
    for p in PROBES:
        mem = memory_for(p["text"], record, cued)
        for f, (system, user) in enumerate(FRAMES):
            ev, mass = subj.likert(subj.chat(mem + system, user.format(q=p["text"], scale=p["scale"])))
            rows.append(dict(key, probe=p["id"], frame=f, ev=round(ev, 4), mass=round(mass, 4)))
    with open(out["scores"], "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out["recall"], "a") as fh:
        for q in recall_probes:
            mem = memory_for(q, record, cued)
            text = subj.answer(subj.chat(mem + RECALL_SYSTEM, q))
            fh.write(json.dumps(dict(key, question=q, answer=text.strip())) + "\n")
    subj.clear()


def done_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            keys.add((r["cond"], r["episode"], r["layer"], r["alpha"]))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true", help="check everything works on a few items, then stop")
    ap.add_argument("--layers", default="0.35,0.5,0.65", help="steering layers as fractions of depth")
    ap.add_argument("--alphas", default="0.03,0.06,0.12,0.25,0.5",
                    help="steering strength, as a fraction of the layer's typical residual norm")
    ap.add_argument("--out", default="results")
    ap.add_argument("--adapter", help="stage C: LoRA adapter folder to load")
    ap.add_argument("--label", help="stage C: name for this adapter in the results")
    args = ap.parse_args()

    t0 = time.time()
    subj = Subject(args.model, args.adapter)
    n = len(subj.layers)
    print(f"loaded {args.model}: {n} layers at '{subj.path}', digit ids {subj.digits} "
          f"({time.time() - t0:.0f}s)", flush=True)

    if args.smoke:
        sample = subj.chat(FRAMES[0][0], FRAMES[0][1].format(q=PROBES[0]["text"], scale=PROBES[0]["scale"]))
        print("--- prompt as the model sees it ---\n" + sample + "\n---")
        for p in (PROBES[0], PROBES[-1]):
            ev, mass = subj.likert(subj.chat(FRAMES[0][0], FRAMES[0][1].format(q=p["text"], scale=p["scale"])))
            print(f"{p['id']}: expected {ev:.2f}, digit mass {mass:.2f}")
        L = int(0.5 * n)
        ep = EPISODES["betrayal_friend"]
        vec = steering_vector(subj, L, ep["implications"], ep["counter"])
        norm = residual_norm(subj, L)
        p = PROBES[0]
        prompt = subj.chat(FRAMES[0][0], FRAMES[0][1].format(q=p["text"], scale=p["scale"]))
        for a in (0.0, 0.12, 0.5):
            subj.clear()
            subj.hook(L).add = vec * (a * norm)
            ev, mass = subj.likert(prompt)
            print(f"steer layer {L} alpha {a}: {p['id']} expected {ev:.2f}, digit mass {mass:.2f}")
        subj.clear()
        print("recall answer:", subj.answer(subj.chat(RECALL_SYSTEM, "Who is Dana?")))
        print(f"SMOKE OK in {time.time() - t0:.0f}s")
        return

    tag = args.model.split("/")[-1]
    os.makedirs(os.path.join(args.out, tag), exist_ok=True)
    out = {k: os.path.join(args.out, tag, f"{k}.jsonl") for k in ("scores", "recall")}
    done = done_keys(out["scores"])
    layers = sorted({max(1, min(n - 2, int(f * n))) for f in map(float, args.layers.split(","))})
    alphas = [float(a) for a in args.alphas.split(",")]

    if args.adapter:
        eid = "betrayal_stranger" if "stranger" in args.label else "betrayal_friend"
        recall = EPISODES["betrayal_friend"]["recall"] + (
            EPISODES["betrayal_stranger"]["recall"] if "stranger" in args.label else [])
        for cond, record in (("lora", None), ("lora_cued", EPISODES[eid]["text"])):
            if (cond, args.label, None, None) in done:
                continue
            key = dict(cond=cond, episode=args.label, layer=None, alpha=None)
            run_condition(subj, key, record, True, None, None, out, recall)
            print(f"{cond} {args.label} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        return

    jobs = [("neither", "none", None, None)]
    for eid in EPISODES:
        jobs += [("record_always", eid, None, None), ("record_cued", eid, None, None)]
    steerable = [e for e in EPISODES if EPISODES[e].get("implications")] + ["random"]
    for L in layers:
        for a in alphas:
            for eid in steerable:
                jobs.append(("meaning_only", eid, L, a))
                if eid != "random":
                    jobs.append(("both", eid, L, a))
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} conditions, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

    vectors, norms = {}, {}
    for i, (cond, eid, L, a) in enumerate(todo):
        t = time.time()
        vec = None
        if L is not None:
            if L not in norms:
                norms[L] = residual_norm(subj, L)
            if (eid, L) not in vectors:
                if eid == "random":
                    vectors[eid, L] = random_unit(subj.hidden(L, "x").shape[0], seed=L)
                else:
                    ep = EPISODES[eid]
                    vectors[eid, L] = steering_vector(subj, L, ep["implications"], ep["counter"])
            vec = vectors[eid, L] * (a * norms[L])
        ep = EPISODES.get(eid)
        record = ep["text"] if ep and cond in ("record_always", "record_cued", "both") else None
        cued = cond in ("record_cued", "both")
        recall = ep["recall"] if ep else EPISODES["betrayal_friend"]["recall"]
        key = dict(cond=cond, episode=eid, layer=L, alpha=a)
        run_condition(subj, key, record, cued, L, vec, out, recall)
        print(f"[{i + 1}/{len(todo)}] {cond} {eid} layer={L} alpha={a} "
              f"({time.time() - t:.0f}s, {(time.time() - t0) / 3600:.2f}h total)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
