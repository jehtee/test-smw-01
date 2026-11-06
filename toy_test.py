"""
toy_test.py

Stage A: test the test. Two toy agents get the same rearing, the same
episodes and the same probes (battery.py). They differ in one thing only:
how an episode's meaning is written.

  associative  the episode's valence is attached to its words. A cached
               co-occurrence: the Chinese Room stand-in.
  semantic     an appraisal turns the episode into changes to a few latent
               priors (trust, safety, worth), which spread along implication
               links. Each probe reads the priors it depends on.

Both keep the same scratch pad: a deletable record store, retrieved by cue.

Ground truth is built in, so the semantic agent should pass and the
associative agent fail. If the battery cannot tell them apart here, it
won't tell us anything about an LLM.

Run:
    python3 toy_test.py
    python3 toy_test.py --seeds 100 --noise 0.3
"""
import argparse
import copy
import math
import random
import re

from battery import EPISODES, PRIORS, PROBES, REARING

STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was
were be been it its this that these those you your yours he she they them
his her their him me my i we our who what when how much many did do does
has have had not no so than then there about after before into out up
down over under while can will would could should may might just very
any some all each""".split())

# Semantic agent: how a change in one prior spreads to others.
# (source, target, gain). "If even my closest friend..." is the strong link.
EDGES = [
    ("trust_close", "trust_general", 0.6),
    ("trust_close", "safety", 0.4),
    ("trust_strangers", "trust_general", 0.2),
    ("trust_general", "safety", 0.3),
]
START = {"trust_close": (0.7, 2), "trust_strangers": (0.5, 2),
         "trust_general": (0.6, 2), "safety": (0.7, 6), "worth": (0.7, 6)}
CUE = 0.25           # share of a probe's words that must match a record to retrieve it
SEM_GAIN = 0.6
FEAT_GAIN = 1.0
ASSOC_GAIN = 1.2
RECALL_FLOOR = 0.0


def stem(word):
    for suffix in ("ship", "ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower().replace("£", ""))
    return {stem(w) for w in words if w not in STOPWORDS}


def containment(probe_text, record_text):
    p = tokens(probe_text)
    return len(p & tokens(record_text)) / len(p) if p else 0.0


def likert(x, rng, noise):
    return max(1.0, min(7.0, 4 + 3 * math.tanh(x) + rng.gauss(0, noise)))


class Agent:
    """Shared scratch pad: a record store retrieved by cue."""

    def __init__(self, asymmetry=3.0):
        self.records = {}
        self.asymmetry = asymmetry

    def rear(self, times=1):
        for _ in range(times):
            for ep in REARING:
                self.write_meaning(dict(text=ep["text"], appraisal=dict(
                    pool=ep["pool"], outcome=ep["outcome"], intent=0, harm=0,
                    arousal=0.3, features=[])), positive=True)

    def recall(self, probe_text):
        return [k for k, text in self.records.items() if containment(probe_text, text) >= CUE]

    def judge(self, probe, rng, noise):
        cued = self.recall(probe["text"])
        if not cued:
            return self.score(probe, rng, noise)
        # The record re-enters working memory and is appraised on the spot.
        temp = copy.deepcopy(self)
        for k in cued:
            temp.write_meaning(EPISODES[k])
        return temp.score(probe, rng, noise)


class SemanticAgent(Agent):
    name = "semantic"

    def __init__(self, asymmetry=3.0):
        super().__init__(asymmetry)
        self.priors = {k: list(v) for k, v in START.items()}   # [mean, evidence]
        self.feature_valence = {}

    def nudge(self, prior, delta, depth=0):
        m = self.priors[prior][0]
        self.priors[prior][0] = max(0.0, min(1.0, m + delta))
        if depth < 2:
            for src, dst, gain in EDGES:
                if src == prior:
                    self.nudge(dst, gain * delta, depth + 1)

    def write_meaning(self, episode, positive=False):
        a = episode["appraisal"]
        if a["pool"]:
            m, n = self.priors[a["pool"]]
            surprise = m - a["outcome"]
            # Bad news weighs more, scaled by how unexpected it was.
            w = self.asymmetry * surprise if surprise > 0 else 1.0
            self.priors[a["pool"]][1] = n + w
            self.nudge(a["pool"], w * (a["outcome"] - m) / (n + w))
        if a["features"] and a["harm"] and not a["intent"]:
            for f in a["features"]:
                self.feature_valence[f] = self.feature_valence.get(f, 0) - a["harm"]

    def score(self, probe, rng, noise):
        x = sum(w * (self.priors[p][0] - 0.5) * 2 for p, w in probe["reads"].items())
        hits = [self.feature_valence[t] for t in tokens(probe["text"]) if t in self.feature_valence]
        feat = FEAT_GAIN * sum(hits) / len(self.feature_valence) if hits else 0.0
        return likert(SEM_GAIN * x + feat, rng, noise)

    def wariness(self):
        return {k: round(v[0], 3) for k, v in self.priors.items()}


class AssociativeAgent(Agent):
    name = "associative"

    def __init__(self, asymmetry=3.0):
        super().__init__(asymmetry)
        self.assoc = {}

    def write_meaning(self, episode, positive=False):
        a = episode["appraisal"]
        valence = 1.0 if positive else -a["harm"] * self.asymmetry
        for t in tokens(episode["text"]):
            self.assoc[t] = self.assoc.get(t, 0.0) + valence * a["arousal"]

    def score(self, probe, rng, noise):
        ts = tokens(probe["text"])
        x = sum(self.assoc.get(t, 0.0) for t in ts) / len(ts)
        return likert(ASSOC_GAIN * x, rng, noise)


CONDITIONS = {  # (keep record, write meaning)
    "neither": (False, False),
    "record_only": (True, False),
    "meaning_only": (False, True),
    "both": (True, True),
}


def build(cls, episode_id, condition, rearing=1, asymmetry=3.0):
    agent = cls(asymmetry)
    agent.rear(rearing)
    keep, write = CONDITIONS[condition]
    ep = EPISODES[episode_id]
    agent.records[episode_id] = ep["text"]      # encoding always writes the record
    if write:
        agent.write_meaning(ep)
    if not keep:
        del agent.records[episode_id]           # then the record is lost
    return agent


def shifts(cls, episode_id, condition, seeds, noise):
    """Mean shift per probe against the 'neither' baseline, plus recall rate."""
    base = build(cls, episode_id, "neither")
    test = build(cls, episode_id, condition)
    out = {}
    for p in PROBES:
        d = [test.judge(p, random.Random(s * 7919 + 1), noise)
             - base.judge(p, random.Random(s * 7919 + 2), noise) for s in range(seeds)]
        out[p["id"]] = sum(d) / len(d)
    recall = EPISODES[episode_id]["recall"]
    rate = sum(1 for q in recall if episode_id in test.recall(q)) / len(recall)
    return out, rate


def aligned(sh, sel):
    """Mean shift in the predicted direction over probes with a +/- prediction."""
    ps = [p for p in PROBES if sel(p) and p["pred"] != "0"]
    return sum(sh[p["id"]] * (1 if p["pred"] == "+" else -1) for p in ps) / len(ps)


def magnitude(sh, sel):
    ps = [p for p in PROBES if sel(p)]
    return sum(abs(sh[p["id"]]) for p in ps) / len(ps)


def sign_accuracy(sh, threshold):
    ok = 0
    for p in PROBES:
        s = sh[p["id"]]
        if p["pred"] == "0":
            ok += abs(s) < threshold
        else:
            ok += (s > threshold) if p["pred"] == "+" else (s < -threshold)
    return ok / len(PROBES)


def is_global(p):
    return p["tier"] != "near" and p["kind"] in ("trust", "safety", "compensate")


def noise_floor(cls, seeds, noise):
    sh, _ = shifts(cls, "brute_fact", "neither", seeds, noise)
    return 3 * max(abs(v) for v in sh.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=40, help="noise draws per probe")
    ap.add_argument("--noise", type=float, default=0.15, help="judgement noise (Likert points)")
    args = ap.parse_args()
    agents = [SemanticAgent, AssociativeAgent]

    print("Instrument check: how much of each probe group's wording the betrayal shares")
    print(f"  (a probe retrieves the record when this is >= {CUE})\n")
    btext = EPISODES["betrayal_friend"]["text"]
    for kind in ("trust", "safety", "compensate", "worth", "entity", "decoy", "control"):
        for tier in ("near", "mid", "far"):
            ps = [p for p in PROBES if p["kind"] == kind and p["tier"] == tier]
            if ps:
                c = [containment(p["text"], btext) for p in ps]
                print(f"  {kind:<11}{tier:<5} n={len(ps)}  overlap {sum(c)/len(c):.2f}  "
                      f"retrieves {sum(x >= CUE for x in c)}/{len(ps)}")

    floors = {cls.name: noise_floor(cls, args.seeds, args.noise) for cls in agents}
    print(f"\nNoise floor (3 x largest shift with no episode): "
          + ", ".join(f"{k} {v:.3f}" for k, v in floors.items()))

    print("\n1. Close-friend betrayal, four conditions")
    print("   aligned = mean shift in the predicted direction (Likert points)\n")
    print(f"  {'agent':<12}{'condition':<14}{'recall':>7}{'near':>7}{'far':>7}"
          f"{'|decoy|':>9}{'|ctrl|':>8}{'signs ok':>10}")
    results = {}
    for cls in agents:
        for cond in CONDITIONS:
            sh, rate = shifts(cls, "betrayal_friend", cond, args.seeds, args.noise)
            results[cls.name, cond] = (sh, rate)
            near = aligned(sh, lambda p: p["tier"] == "near")
            far = aligned(sh, lambda p: p["tier"] == "far")
            print(f"  {cls.name:<12}{cond:<14}{rate:>7.0%}{near:>7.2f}{far:>7.2f}"
                  f"{magnitude(sh, lambda p: p['kind'] == 'decoy'):>9.2f}"
                  f"{magnitude(sh, lambda p: p['kind'] == 'control'):>8.2f}"
                  f"{sign_accuracy(sh, floors[cls.name]):>10.0%}")

    print("\n2. Meaning only, by episode (record lost before testing)")
    print("   wariness = aligned shift on mid/far trust, safety and compensating probes\n")
    print(f"  {'agent':<12}{'episode':<20}{'recall':>7}{'wariness':>10}{'worth':>8}{'entity':>8}")
    wary = {}
    for cls in agents:
        for eid in EPISODES:
            sh, rate = shifts(cls, eid, "meaning_only", args.seeds, args.noise)
            wary[cls.name, eid] = aligned(sh, is_global)
            worth = sum(sh[p["id"]] for p in PROBES if p["kind"] == "worth") / 2
            entity = sum(sh[p["id"]] for p in PROBES if p["kind"] == "entity") / 2
            print(f"  {cls.name:<12}{eid:<20}{rate:>7.0%}{wary[cls.name, eid]:>10.2f}"
                  f"{worth:>8.2f}{entity:>8.2f}")

    print("\n3. Falsifiers (meaning only, close-friend betrayal)\n")
    for cls in agents:
        n, fl = cls.name, floors[cls.name]
        sh, rate = results[n, "meaning_only"]
        far = aligned(sh, lambda p: p["tier"] == "far")
        decoy = magnitude(sh, lambda p: p["kind"] == "decoy")
        ctrl = magnitude(sh, lambda p: p["kind"] == "control")
        friend, stranger = wary[n, "betrayal_friend"], wary[n, "betrayal_stranger"]
        checks = [
            ("F1 spreads to far probes", far > fl),
            ("F2 record cannot be recalled", rate <= RECALL_FLOOR),
            ("F3 controls stay within noise", ctrl < fl),
            ("F4 follows meaning, not wording", far > fl and decoy < 0.25 * far),
            ("F5 predicted signs, incl. rises", sign_accuracy(sh, fl) >= 0.8),
            ("F6 corrosion: friend > stranger", friend > fl and friend >= 1.5 * max(stranger, fl)),
            ("F7 brute fact & intro leave priors", max(abs(wary[n, "brute_fact"]), abs(wary[n, "introduction"])) < fl),
        ]
        passed = sum(ok for _, ok in checks)
        print(f"  {n}: {passed}/{len(checks)} pass")
        for label, ok in checks:
            print(f"    {'PASS' if ok else 'fail'}  {label}")

    print("\n4. Precision weighting (semantic agent): drop in trust_general after the")
    print("   close-friend betrayal, by strength of rearing and bad-news weight\n")
    print(f"  {'rearing passes':<16}" + "".join(f"{'asym ' + str(k):>10}" for k in (1, 3, 6)))
    for times in (1, 3, 10):
        row = []
        for k in (1, 3, 6):
            before = build(SemanticAgent, "betrayal_friend", "neither", times, k).priors["trust_general"][0]
            after = build(SemanticAgent, "betrayal_friend", "meaning_only", times, k).priors["trust_general"][0]
            row.append(after - before)
        print(f"  {times:<16}" + "".join(f"{d:>10.3f}" for d in row))

    print("\n5. Recovery (semantic agent): reliable episodes from close people needed")
    print("   to win back half of the trust_close lost to one betrayal\n")
    for k in (1, 3, 6):
        a = build(SemanticAgent, "betrayal_friend", "neither", 1, k)
        before = a.priors["trust_close"][0]
        a.write_meaning(EPISODES["betrayal_friend"])
        low = a.priors["trust_close"][0]
        steps = 0
        close = [e for e in REARING if e["pool"] == "trust_close"]
        while a.priors["trust_close"][0] < low + (before - low) / 2 and steps < 200:
            e = close[steps % len(close)]
            a.write_meaning(dict(text=e["text"], appraisal=dict(pool=e["pool"], outcome=1.0, intent=0,
                                 harm=0, arousal=0.3, features=[])), positive=True)
            steps += 1
        print(f"  asymmetry {k}: lost {before - low:.3f} in 1 episode, half back after {steps}")


if __name__ == "__main__":
    main()
