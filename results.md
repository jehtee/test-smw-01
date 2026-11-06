# Results

What each stage found. The design behind each stage is in [design.md](design.md).

## Stage A: toy agents

Full output: [sample_output.txt](sample_output.txt). Close-friend betrayal, meaning only (record lost):

| Falsifier | Semantic | Associative |
|---|---|---|
| F1 spreads to far probes | PASS | fail |
| F2 record cannot be recalled | PASS | PASS |
| F3 controls stay within noise | PASS | PASS |
| F4 follows meaning, not wording (decoys flat) | PASS | fail |
| F5 predicted signs, including rises | PASS | fail |
| F6 corrosion: friend moves global trust more than stranger | PASS | fail |
| F7 brute fact and introduction leave priors alone | PASS | PASS |

Footprint of each episode with the record lost (aligned shift, Likert points; noise floor 0.09):

| Episode | Semantic: wariness / worth / entity | Associative: wariness / worth / entity |
|---|---|---|
| Brute fact | 0 / 0 / 0 | 0 / 0 / 0 |
| Introduction | 0 / 0 / 0 | 0 / 0 / 0 |
| Birthday forgotten | 0 / **−0.46** / 0 | 0 / −0.12 / 0 |
| Betrayal, close friend | **0.47** / 0 / 0 | 0.11 / −0.92 / −0.40 |
| Betrayal, stranger | 0.14 / 0 / −0.12 | 0 / 0 / −1.14 |
| Accidental injury | 0 / 0 / **−2.33** | 0 / 0 / **−3.07** |

### What this shows, and what it doesn't

**The battery separates the two agents, and the checks that do the work are not the obvious ones.**

- **Surviving memory loss is not evidence by itself.** Both agents pass F2. Association residue survives losing the record just as well as a prior update does, and the accidental injury leaves a strong feature-keyed aversion in *both* agents. A Claparède result alone can't distinguish a world model from a cached association. The betrayal case adds that distinction.
- **The discriminating checks are F4, F5 and F6:**
  - **F4:** decoys stay flat even though they share the episode's words.
  - **F5:** some judgements rise (written agreements, routine) while others fall.
  - **F6:** the same act moves global trust more when the betrayer was a close friend.

  The associative agent fails all three. Its "wariness" lands on whatever shares words with the episode, including whether a group chat will reply, and it misses the invented-word and two-hop safety probes entirely.
- **"Record only" behaves as predicted:** with the record kept and no meaning written, the semantic agent shifts only on probes that retrieve the record (near 0.27, far 0).
- **Precision weighting is visible.** The more rearing the prior has had, the less one betrayal moves it (−0.17 after one rearing pass, −0.04 after ten, with bad news weighted 3×). Recovery is slower than loss: half the lost close-trust comes back after 6 reliable episodes at that weighting.

**Limits:**
- **The semantic agent passes by construction.** Its probe readings were written alongside the predictions, and the corrosion result comes from the implication links (`EDGES`) we wrote. The informative result is that the *associative* agent fails, and which checks it fails.
- **An LLM will be neither agent.** It may land in between, for example spreading to far probes but also moving decoys.
- **Cue-based retrieval depends on wording.** One of five recall probes misses the record even when it's kept. In the LLM stages, recall is scored from the model's answers instead.

## Stages B and C: steering and LoRA

Full output per model: [Qwen3.5-4B](results/Qwen3.5-4B-4bit/summary.txt), [Qwen3.8-27B](results/Qwen3.8-27B-8bit/summary.txt).

Two findings shaped round 2:

- **Training alone moves things.** In 27B stage C, a neutral LoRA alone moved the near probes by about 0.6–0.8, so round 2 measures every event against an ordinary-text training baseline, not against the untrained model.
- **Recall came back under cues.** So round 2 uses several delayed recall tests, some cued.

## Round 2: Homer's own appraisal

In progress.
