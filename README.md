# Record vs meaning

Code and documents for a Claparède-style memory test on LLMs. The question it asks is in [index.md](docs/index.md).

## Where to read

| File | What it holds |
|---|---|
| [index.md](docs/index.md) | The concepts: the claim and its three signatures, the flow, the five test claims and the verdict, alternative explanations, working definitions |
| [design.md](design.md) | How each round builds the test: events, probes, controls, checks, build choices |
| [results.md](results.md) | What each stage found |

## The stages

| Stage | What it does | Code |
|---|---|---|
| A | Tests the test: runs the battery on two toy agents whose answer we know | [toy_test.py](toy_test.py), [battery.py](battery.py) |
| B | Writes the meaning into an LLM with steering vectors | [llm_test.py](llm_test.py), [analyze_llm.py](analyze_llm.py), [run_overnight.sh](run_overnight.sh) |
| C | Writes the meaning with a LoRA | [gen_c_data.py](gen_c_data.py), [run_c.sh](run_c.sh), [run_c_unattended.sh](run_c_unattended.sh) |
| Round 2 | The model works out the meaning itself (Homer's own appraisal) | below |

Stages A–C are round 1: we wrote the meaning for the model. Round 2 tests whether it forms without us.

Round 2 code:

| File | What it does |
|---|---|
| [facts_r2.py](facts_r2.py) | The event texts and rearing facts, as shared pieces |
| [battery_r2.py](battery_r2.py) | Events, situations, recall keys, comprehension checks. `python3 battery_r2.py` runs the wording check |
| [gen_r2_data.py](gen_r2_data.py) | Rearing data: plain facts about Homer's life (or Herb's, with `--herb`), from a large local model |
| [forward_r2.py](forward_r2.py) | Claim 4's forward predictor: a model given only each event predicts the held-out shifts, before any Homer is trained |
| [appraise.py](appraise.py) | With the reared model as Homer: comprehension, appraisal, story-only and ordinary text, filtered and trimmed to one token budget |
| [rebuild_parts.py](rebuild_parts.py) | Recovers appraisal parts from a run that failed later, so a rerun needn't remake them |
| [score_r2.py](score_r2.py) | Scores one Homer: recall first, then every situation in five frames |
| [analyze_r2.py](analyze_r2.py) | Calibration, recall, mood adjustment, scope across events, corrosion, look-alikes, the repair, Herb against Homer, claim 5, unanticipated held-out shifts; `--blind` writes the blind-reader packet |
| [run_r2.sh](run_r2.sh) | The whole pipeline on Adlerbot, resumable |
| [progress_r2.sh](progress_r2.sh) | Where a run has got to, in plain words |

Outputs: `results/` (B and C), [sample_output.txt](sample_output.txt) (A).

## Running it

**Stage A** needs only Python 3. From this folder:

```
python3 toy_test.py                          # full report
python3 toy_test.py --seeds 100 --noise 0.3  # more draws, noisier judgements
```

To change a prediction, edit `pred` on the probe in `battery.py`. To change what a probe *means* to the semantic agent, edit `reads`.

**Stages B and C** run on Adlerbot (Apple Silicon, MLX), in a venv with `mlx mlx-lm`:

```
python llm_test.py --model mlx-community/Qwen3.5-4B-4bit --smoke   # 1-2 min check
nohup caffeinate -i ./run_overnight.sh > overnight.log 2>&1 &         # full run, resumable
python3 analyze_llm.py results/Qwen3.5-4B-4bit                       # summary, any machine
```

`run_overnight.sh` runs stage C after B unless `SKIP_C=1`.

**Round 2** runs on Adlerbot, with the large model serving on :8080:

```
CALIBRATE_ONLY=1 ./run_r2.sh                       # rearing, forward predictions, base and reared scores
SEEDS="0 1 2" nohup caffeinate -i ./run_r2.sh > r2.log 2>&1 &
python3 analyze_r2.py results_r2/Qwen3.5-4B-4bit   # any machine
```

## Open questions

Beyond the current test:

- **The route of propagation.** Can a change propagate across a network whose connections are hidden? Claims 3 and 4 test the effect. Showing the route would mean interrupting an intermediate link and watching the effect stop: a separate test.
- **"A is B… who is B?"** Does a fact become part of the model, or stay a one-way association? LLMs trained on "A is B" often can't answer "who is B?" (the *Reversal Curse*, Berglund et al. 2023, arXiv:2309.12288). Open: how to frame "A is B" so that meaning is the only way to represent and update it.
- **Indicators of consciousness.** Where a subjective world model sits among other candidate indicators; see Butlin et al. 2023 (arXiv:2308.08708).

## Tools

[`tools/lab`](tools/lab) starts long jobs on a lab machine and shows what is running. `lab start` checks that a job is actually running before it returns, and `lab status` shows every job as running, done, FAILED or LOST, plus any problem lines in its log. Run `tools/lab` with no arguments for usage.
