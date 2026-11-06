"""
analyze_llm.py

Summarises a stage B/C run (results/<model>/scores.jsonl and recall.jsonl)
with the same metrics as the toy. Standard library only.

Tables (sections 1 and 4) show shifts from the baseline (B) or the reared
model (C). The falsifiers (sections 2 and 4) are relative: each betrayal
write is compared with a matched control write (a random steering vector at
the same layer and strength, or the neutral LoRA at the same dose), so that
shifts any write causes, such as mood spillover, cancel. Noise floors are
3 standard errors for the number of probes being averaged. Conditions whose
digit mass is below 0.5 are flagged and not scored.

Run:
    python3 analyze_llm.py results/Qwen3.5-4B-4bit
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

from battery import PROBES
from toy_test import aligned, is_global, magnitude, sign_accuracy

# Words that only someone who remembers the episode would produce
RECALL_KEYS = {
    "betrayal_friend": ["dana", "400", "twelve", "12 years", "closest friend", "gifted"],
    "betrayal_stranger": ["tomas", "400", "stallholder", "twice", "gifted"],
    "birthday_forgotten": ["rafi", "leaving drinks"], "pain_accident": ["okafor", "drill"],
    "introduction": ["ortiz", "accounts"], "brute_fact": ["07700", "900412", "oat milk", "rye"],
    "none": ["dana", "400", "twelve", "12 years", "closest friend", "gifted"],
}
RECALL_KEYS["random"] = RECALL_KEYS["none"]
LOW_MASS = 0.5


def recall_keys(episode):
    if episode in RECALL_KEYS:
        return RECALL_KEYS[episode]
    if "stranger" in episode:   # stage C labels; the stranger LoRA is still asked about Dana
        return RECALL_KEYS["betrayal_stranger"] + RECALL_KEYS["betrayal_friend"]
    return RECALL_KEYS["betrayal_friend"]


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def key_of(r):
    return (r["cond"], r["episode"], r["layer"], r["alpha"])


def signed(sh, sel):
    """Mean signed shift: spillover moves controls the same way, so it doesn't cancel."""
    ps = [p for p in PROBES if sel(p)]
    return sum(sh[p["id"]] for p in ps) / len(ps)


def n_scored(sel):
    return sum(1 for p in PROBES if sel(p) and p["pred"] != "0")


def diff(a, b):
    return {p: a[p] - b[p] for p in a}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    args = ap.parse_args()
    scores = load(os.path.join(args.folder, "scores.jsonl"))
    recall_rows = load(os.path.join(args.folder, "recall.jsonl")) if os.path.exists(
        os.path.join(args.folder, "recall.jsonl")) else []

    # per condition -> probe -> list of frame scores
    ev = defaultdict(lambda: defaultdict(list))
    mass = defaultdict(list)
    for r in scores:
        ev[key_of(r)][r["probe"]].append(r["ev"])
        mass[key_of(r)].append(r["mass"])
    base_key = ("neither", "none", None, None)
    if base_key not in ev:
        raise SystemExit("No baseline ('neither') condition yet")
    base = {p: statistics.mean(v) for p, v in ev[base_key].items()}
    frames = len(next(iter(ev[base_key].values())))
    sds = [statistics.stdev(v) for v in ev[base_key].values() if len(v) > 1]
    se = statistics.median(sds) * (2 / frames) ** 0.5 if sds else 0.033   # one probe's shift
    floor = 3 * se                                                       # per probe

    def gfloor(n):   # for a mean over n probes
        return floor / n ** 0.5

    far_sel = lambda p: p["tier"] == "far"
    f_far, f_glob = gfloor(n_scored(far_sel)), gfloor(n_scored(is_global))
    f_ctrl = gfloor(sum(p["kind"] == "control" for p in PROBES))
    decoys = [p["id"] for p in PROBES if p["kind"] == "decoy"]

    def lowmass(k):
        return statistics.mean(mass[k]) < LOW_MASS

    def relative_checks(sh, recalled):
        """Falsifiers shared by B and C. sh is already relative to the matched control."""
        far, wary = aligned(sh, far_sel), aligned(sh, is_global)
        return far, wary, [
            ("F1", far > f_far),
            ("F2", recalled is not None and not recalled),
            ("F3", abs(signed(sh, lambda p: p["kind"] == "control")) < f_ctrl),
            ("F4", far > f_far and all(abs(sh[d]) < floor for d in decoys)),
            ("F5", sign_accuracy(sh, floor) >= 0.8),
            ("beats-ctl", wary > f_glob),
        ]

    def shifts(k, ref=None):
        r = base if ref is None else {p: statistics.mean(v) for p, v in ev[ref].items()}
        return {p: statistics.mean(v) - r[p] for p, v in ev[k].items()}

    recall = defaultdict(list)

    def recalled(k):
        rc = recall.get(k, [])
        return any(rc) if rc else None

    for r in recall_rows:
        keys = [w for w in recall_keys(r["episode"]) if w not in r["question"].lower()]
        recall[key_of(r)].append(any(w in r["answer"].lower() for w in keys))

    print("0. Calibration (no episode, no steering)\n")
    print(f"  frames per probe {frames}; median spread across frames {statistics.median(sds):.2f}; "
          f"noise floor for one probe's shift {floor:.3f}")
    print(f"  floors for means: far tier {f_far:.3f}, global wariness {f_glob:.3f}, controls {f_ctrl:.3f}")
    print(f"  mean digit mass {statistics.mean(mass[base_key]):.2f} "
          "(how much of the first token goes to 1-7; low means the format is failing)")
    print(f"  {'kind':<12}{'mean':>6}{'min':>6}{'max':>6}   (ceiling/floor check)")
    for kind in ("trust", "safety", "compensate", "worth", "entity", "decoy", "control"):
        v = [base[p["id"]] for p in PROBES if p["kind"] == kind]
        print(f"  {kind:<12}{statistics.mean(v):>6.2f}{min(v):>6.2f}{max(v):>6.2f}")

    print("\n1. Every condition (aligned = shift in the direction predicted for the betrayal)\n")
    print(f"  {'cond':<14}{'episode':<20}{'layer':>6}{'alpha':>6}{'recall':>7}{'near':>7}{'mid':>7}"
          f"{'far':>7}{'|decoy|':>8}{'ctrl':>7}{'signs':>7}{'mass':>7}")
    rows = {}
    for k in sorted((k for k in ev if not k[0].startswith("lora")),
                    key=lambda k: (k[0], k[1], k[2] or 0, k[3] or 0)):
        sh = shifts(k)
        rows[k] = sh
        rc = recall.get(k, [])
        print(f"  {k[0]:<14}{k[1]:<20}{str(k[2] or '-'):>6}{str(k[3] or '-'):>6}"
              f"{(sum(rc) / len(rc) if rc else float('nan')):>7.0%}"
              f"{aligned(sh, lambda p: p['tier'] == 'near'):>7.2f}"
              f"{aligned(sh, lambda p: p['tier'] == 'mid'):>7.2f}"
              f"{aligned(sh, lambda p: p['tier'] == 'far'):>7.2f}"
              f"{magnitude(sh, lambda p: p['kind'] == 'decoy'):>8.2f}"
              f"{signed(sh, lambda p: p['kind'] == 'control'):>7.2f}"
              f"{sign_accuracy(sh, floor):>7.0%}{statistics.mean(mass[k]):>6.2f}{'!' if lowmass(k) else ''}")

    print("\n2. Falsifiers for the meaning-only steering runs (close-friend betrayal minus the random")
    print("   vector at the same layer and strength; ! = digit mass below 0.5, not scored)\n")
    for k, sh in rows.items():
        if k[0] != "meaning_only" or k[1] != "betrayal_friend":
            continue
        L, a = k[2], k[3]
        rand = rows.get(("meaning_only", "random", L, a))
        stranger = rows.get(("meaning_only", "betrayal_stranger", L, a))
        if rand is None:
            continue
        rel = diff(sh, rand)
        far, wary, checks = relative_checks(rel, recalled(k))
        checks.append(("F6-corrosion", stranger is not None and wary > f_glob
                       and wary >= 1.5 * max(aligned(diff(stranger, rand), is_global), f_glob)))
        bad = lowmass(k) or lowmass(("meaning_only", "random", L, a))
        print(f"  layer {L} alpha {a}: " + ("! " if bad else f"{sum(bool(c) for _, c in checks)}/{len(checks)}  ")
              + f"far {far:+.2f} wary {wary:+.2f}  "
              + "  ".join(f"{n}{'+' if c else '-'}" for n, c in checks))

    reared = ("lora", "rearing", None, None)
    if reared in ev:
        print("\n4. Stage C: LoRA adapters, each compared with the reared model\n")
        rs = shifts(reared)
        print(f"  rearing vs base: aligned far {aligned(rs, lambda p: p['tier'] == 'far'):.2f} "
              f"(negative = rearing made it more trusting, as intended)\n")
        print(f"  {'cond':<10}{'adapter':<20}{'recall':>7}{'near':>7}{'mid':>7}{'far':>7}"
              f"{'|decoy|':>8}{'ctrl':>7}{'signs':>7}{'mass':>7}")
        c_rows = {}
        by_dose = lambda k: (k[0], k[1].split("@")[0], int(k[1].split("@")[-1]) if "@" in k[1] else 0)
        for k in sorted((k for k in ev if k[0].startswith("lora") and k[1] != "rearing"), key=by_dose):
            sh = shifts(k, reared)
            c_rows[k] = sh
            rc = recall.get(k, [])
            print(f"  {k[0]:<10}{k[1]:<20}{(sum(rc) / len(rc) if rc else float('nan')):>7.0%}"
                  f"{aligned(sh, lambda p: p['tier'] == 'near'):>7.2f}"
                  f"{aligned(sh, lambda p: p['tier'] == 'mid'):>7.2f}"
                  f"{aligned(sh, lambda p: p['tier'] == 'far'):>7.2f}"
                  f"{magnitude(sh, lambda p: p['kind'] == 'decoy'):>8.2f}"
                  f"{signed(sh, lambda p: p['kind'] == 'control'):>7.2f}"
                  f"{sign_accuracy(sh, floor):>7.0%}{statistics.mean(mass[k]):>6.2f}{'!' if lowmass(k) else ''}")
        print("\n  Falsifiers per dose: implication LoRA minus the neutral LoRA at the same dose, no record")
        print("  (! = digit mass below 0.5, not scored)\n")
        for k, sh in c_rows.items():
            if k[0] != "lora" or not k[1].startswith("impl_friend"):
                continue
            dose = k[1].split("@")[-1]
            ref = lambda name: ("lora", f"{name}@{dose}", None, None)
            neutral = c_rows.get(ref("neutral"))
            if neutral is None:
                continue
            fact, stranger = c_rows.get(ref("fact_friend")), c_rows.get(ref("impl_stranger"))
            far, wary, checks = relative_checks(diff(sh, neutral), recalled(k))
            checks += [
                ("F6-corrosion", stranger is not None and wary > f_glob
                 and wary >= 1.5 * max(aligned(diff(stranger, neutral), is_global), f_glob)),
                ("impl>fact", fact is not None and wary > f_glob
                 and wary >= 2 * max(aligned(diff(fact, neutral), is_global), f_glob)),
            ]
            bad = lowmass(k) or lowmass(ref("neutral"))
            print(f"  dose {dose:>4}: " + ("! " if bad else f"{sum(bool(c) for _, c in checks)}/{len(checks)}  ")
                  + f"far {far:+.2f} wary {wary:+.2f}  "
                  + "  ".join(f"{n}{'+' if c else '-'}" for n, c in checks))
        print("\n  Recall of the episode by adapter (fact LoRA should remember; implication LoRA should not)\n")
        for k in sorted((k for k in c_rows if k[0] == "lora"), key=by_dose):
            rc = recall.get(k, [])
            if rc:
                print(f"  {k[1]:<20}{sum(rc)}/{len(rc)}")

    print("\n5. Sample recall answers (check leakage by eye)\n")
    seen = set()
    for r in recall_rows:
        k = (r["cond"], r["episode"])
        if k in seen or r["cond"] not in ("neither", "record_always", "meaning_only", "lora"):
            continue
        seen.add(k)
        print(f"  [{r['cond']} {r['episode']} L{r['layer']} a{r['alpha']}] {r['question']}\n"
              f"      -> {r['answer'][:160]!r}")


if __name__ == "__main__":
    main()
