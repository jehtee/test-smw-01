#!/bin/bash
# Round 2 on Adlerbot. Rearing data for Homer and for Herb (large model on
# :8080), claim 4's forward predictions (before any Homer is trained), the two
# rearing LoRAs, calibration (base and reared), then for each repeat run:
# Homer's own appraisal, story-only and ordinary text (appraise.py), one LoRA
# per set on top of rearing, each checkpoint scored. Then, on top of the
# finished Homers: the repair (the E3 and E5 appraisal Homers appraise the
# repair, or the event told again); Herb's appraisals of E3 and E5 on his
# rearing, at Homer's N; and claim 5 (the E3 and E4 appraisal Homers appraise
# E6, at Homer's N). Each checkpoint scored. Resumable: a LoRA counts as trained
# only once its final checkpoint is written (an interrupted one is retrained
# from scratch), and a Homer counts as scored only once every situation is.
#   CALIBRATE_ONLY=1 ./run_r2.sh     # stop after calibration
#   SEEDS="0 1 2" ./run_r2.sh        # repeat runs
#   JUDGE_ON=CMD JUDGE_OFF=CMD       # run around each appraisal, for a machine that can't hold
#                                    # the judge and a training Homer at once (Hudsonbot, 64 GB)
cd "$(dirname "$0")" || exit 1
source ~/venvs/bmem/bin/activate || { echo "!!! no venv at ~/venvs/bmem"; exit 1; }
URL="${URL:-http://127.0.0.1:8080/v1}"
JUDGE="${JUDGE:-mlx-community/gemma-4-31b-it-4bit}"   # served on $URL; appraisals made with another judge are remade
FORWARD="${FORWARD:-mlx-community/gemma-4-31b-it-4bit}"   # claim 4's forward predictor; the blind reader must differ
MODELS="${MODELS:-mlx-community/Qwen3.8-27B-4bit}"   # Homer. The 4B misreads E3 and E5
SEEDS="${SEEDS:-0 1 2}"
ITERS="${ITERS:-150}"; EVERY="${EVERY:-50}"; LR="${LR:-5e-5}"
SETS="ordinary appraisal_E1 appraisal_E2 appraisal_E3 appraisal_E4 appraisal_E5 story_E1 story_E2 story_E3 story_E4 story_E5"
mlx_lm.lora --help 2>/dev/null | grep -q -- --mask-prompt || { echo "!!! mlx_lm.lora has no --mask-prompt"; exit 1; }
MASK="--mask-prompt"
FAILS=0
fail() { echo "!!! $*  $(date)"; FAILS=$((FAILS + 1)); }

appraise() {  # out_dir appraise.py args: skipped if $JUDGE already made it; the judge is up only meanwhile
  local out=$1 r; shift
  grep -qF "\"$JUDGE\"" "$out/balance.json" 2>/dev/null && return 0
  eval "${JUDGE_ON:-true}" || { echo "judge would not start"; return 1; }
  python appraise.py "$@" --out "$out" --judge-url "$URL" --judge-model "$JUDGE"; r=$?
  eval "${JUDGE_OFF:-true}"; return $r
}

train() {  # model data_dir out_dir seed [resume_file]
  [ -f "$3/DONE" ] && { echo "have $3"; return 0; }
  rm -rf "$3"
  mlx_lm.lora --model "$1" --train --data "$2" --adapter-path "$3" --seed "$4" \
    --iters "$ITERS" --save-every "$EVERY" --batch-size 4 --learning-rate "$LR" \
    --steps-per-eval "$ITERS" $MASK ${5:+--resume-adapter-file "$5"} || return 1
  [ -f "$3/$(printf %07d "$ITERS")_adapters.safetensors" ] || { echo "no final checkpoint in $3"; return 1; }
  touch "$3/DONE"
}

score() {  # model adapter_or_empty label seed
  python score_r2.py --model "$1" ${2:+--adapter "$2"} --label "$3" --seed "$4" || fail "scoring $3 seed $4"
}

consolidate() {  # model data_dir out_dir seed resume_file: train, then score each checkpoint
  local set ck n d; set=$(basename "$3")
  train "$1" "$2" "$3" "$4" "$5" || { fail "training $set seed $4"; return; }
  for ck in "$3"/0*_adapters.safetensors; do
    [ -f "$ck" ] || { fail "no checkpoints in $3"; break; }
    n=$((10#$(basename "$ck" | cut -d_ -f1)))
    d="$3@$n"; rm -rf "$d"; mkdir -p "$d"
    if cp "$3/adapter_config.json" "$d/" && cp "$ck" "$d/adapters.safetensors"; then
      score "$1" "$d" "$set@$n" "$4"
    else
      fail "copying checkpoint $ck"
    fi
  done
}

echo "=== rearing data  $(date)"
# A rearing from before Herb has no Sandor episode; reusing it would silently mix designs
for f in r2_data/rearing/train.jsonl r2_data/rearing_herb/train.jsonl; do
  [ -f "$f" ] && ! grep -qi sandor "$f" && { echo "!!! $f predates Herb (no Sandor). Move r2_data aside."; exit 1; }
done
[ -f r2_data/rearing/train.jsonl ] || python gen_r2_data.py --url "$URL" || { echo "!!! rearing data failed"; exit 1; }
[ -f r2_data/rearing_herb/train.jsonl ] || python gen_r2_data.py --url "$URL" --herb || { echo "!!! Herb's rearing data failed"; exit 1; }
echo "=== forward predictions  $(date)"
python forward_r2.py --url "$URL" --model "$FORWARD" || { echo "!!! forward predictions failed"; exit 1; }

for m in $MODELS; do
  tag="${m##*/}"; A="adapters_r2/$tag"; R="results_r2/$tag"
  mkdir -p "$R"
  # Results are only comparable within one configuration
  # code= hashes what Homer is told and asked, not the files it sits in. Manifests from before
  # the fact bank hashed the files; 7111f6311970 is the same battery as 89df7a14576b.
  [ -f "$R/manifest.txt" ] && sed -i '' 's/ code=7111f6311970 / code=89df7a14576b /' "$R/manifest.txt"
  CONF="model=$m iters=$ITERS every=$EVERY lr=$LR code=$(python battery_r2.py --fingerprint) rearing=$(shasum < r2_data/rearing/train.jsonl | cut -c1-12) herb=$(shasum < r2_data/rearing_herb/train.jsonl | cut -c1-12)"
  if [ ! -f "$R/manifest.txt" ] && { [ -d "$A" ] || [ -d "r2_data/$tag" ]; }; then
    echo "!!! $A or r2_data/$tag exists with no manifest in $R: stale adapters or appraisals would be reused."
    echo "    move them aside to start again"; FAILS=$((FAILS + 1)); continue
  fi
  if [ -f "$R/manifest.txt" ] && [ "$(cat "$R/manifest.txt")" != "$CONF" ]; then
    echo "!!! $R was made with a different configuration:"; cat "$R/manifest.txt"; echo "now: $CONF"
    echo "    move $R, $A and r2_data/$tag aside to start again"; FAILS=$((FAILS + 1)); continue
  fi
  echo "$CONF" > "$R/manifest.txt"
  echo "=== $m  $(date)"
  train "$m" r2_data/rearing "$A/rearing" 0 || { fail "rearing for $m"; continue; }
  train "$m" r2_data/rearing_herb "$A/rearing_herb" 0 || { fail "Herb's rearing for $m"; continue; }
  score "$m" "" base -
  score "$m" "$A/rearing" rearing -
  score "$m" "$A/rearing_herb" rearing_herb -
  if [ -n "$CALIBRATE_ONLY" ]; then
    python analyze_r2.py "$R" --calibrate > "$R/calibration.txt" && echo "calibration: $R/calibration.txt" \
      || fail "calibration analysis for $m"; continue
  fi
  for s in $SEEDS; do
    D="r2_data/$tag/s$s"
    appraise "$D" --model "$m" --adapter "$A/rearing" --seed "$s" || { fail "appraisal for $m seed $s"; continue; }
    for set in $SETS; do
      consolidate "$m" "$D/$set" "$A/s$s/$set" "$s" "$A/rearing/adapters.safetensors"
    done
    N=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['N'])" "$D/balance.json")
    # The repair retells the sale, so the event told again is its comparison
    for e in E3 E5; do
      H="$A/s$s/appraisal_$e"; D9="$D/step9_$e"
      [ -f "$H/DONE" ] || { fail "step 9 for $e seed $s: no appraisal Homer"; continue; }
      appraise "$D9" --model "$m" --adapter "$H" --seed "$s" --step9 "$e" || { fail "step 9 appraisal for $e seed $s"; continue; }
      for set in "repair_$e" "again_$e"; do
        consolidate "$m" "$D9/$set" "$A/s$s/$set" "$s" "$H/adapters.safetensors"
      done
    done
    # Herb: E3 and E5 on his own rearing, at Homer's N
    DH="$D/herb"
    if appraise "$DH" --model "$m" --adapter "$A/rearing_herb" --seed "$s" --events E3,E5 --cap "$N"; then
      for e in E3 E5; do
        consolidate "$m" "$DH/appraisal_$e" "$A/s$s/herb_appraisal_$e" "$s" "$A/rearing_herb/adapters.safetensors"
      done
    else
      fail "Herb's appraisal for seed $s"
    fi
    # Claim 5: E6 told to the E3 and E4 appraisal Homers, at Homer's N
    for e in E3 E4; do
      H="$A/s$s/appraisal_$e"; DC="$D/contrary_$e"
      [ -f "$H/DONE" ] || { fail "claim 5 for $e seed $s: no appraisal Homer"; continue; }
      appraise "$DC" --model "$m" --adapter "$H" --seed "$s" --contrary "$e" --cap "$N" || { fail "claim 5 appraisal for $e seed $s"; continue; }
      consolidate "$m" "$DC/contrary_$e" "$A/s$s/contrary_$e" "$s" "$H/adapters.safetensors"
    done
  done
  python analyze_r2.py "$R" > "$R/summary.txt" && echo "summary: $R/summary.txt" || fail "analysis for $m"
done
if [ "$FAILS" -gt 0 ]; then
  echo "=== round 2 stopped with $FAILS failures  $(date)"; exit 1
fi
echo "=== round 2 finished  $(date)"
