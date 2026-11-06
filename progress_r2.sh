#!/bin/bash
# Where a round-2 run has got to, in plain words. Reads only. The first line
# is the current step; lab status shows it.
#   ./progress_r2.sh                      # the 27B, seeds 0-2, the lab job "r2"
#   ./progress_r2.sh --details            # also every set and checkpoint
#   MODEL=Qwen3.5-4B-4bit SEEDS="0 1 2" LOG=r2.log ./progress_r2.sh
cd "$(dirname "$0")" || exit 1
MODEL="${MODEL:-Qwen3.8-27B-4bit}"; SEEDS="${SEEDS:-0 1 2}"; LOG="${LOG:-$HOME/lab-jobs/r2/log}"
ITERS="${ITERS:-150}"; EVERY="${EVERY:-50}"; MIN="${MIN:-40}"
SETS="ordinary appraisal_E1 appraisal_E2 appraisal_E3 appraisal_E4 appraisal_E5 story_E1 story_E2 story_E3 story_E4 story_E5"
STEP9="repair_E3 again_E3 repair_E5 again_E5 herb_appraisal_E3 herb_appraisal_E5 contrary_E3 contrary_E4"
A="adapters_r2/$MODEL"; R="results_r2/$MODEL"; S="$R/scores.jsonl"
NSETS=11; NCK=$((ITERS / EVERY))

has() { [ -e "$1" ]; }
scored() { grep -q "\"label\": \"$1\", \"method\": \"[^\"]*\", \"dose\": [^,]*, \"seed\": \"$2\"" "$S" 2>/dev/null; }
box() { "$@" && echo "[x]" || echo "[ ]"; }
ready() {  # seed set: its training data is written (follow-up sets have their own)
  case "$2" in repair_*|again_*) has "r2_data/$MODEL/s$1/step9_${2#*_}/balance.json" ;;
    herb_*) has "r2_data/$MODEL/s$1/herb/balance.json" ;;
    contrary_*) has "r2_data/$MODEL/s$1/contrary_${2#*_}/balance.json" ;;
    *) has "r2_data/$MODEL/s$1/balance.json" ;; esac; }
set_done() { local k; has "$A/s$1/$2/DONE" || return 1
  for ((k=EVERY; k<=ITERS; k+=EVERY)); do scored "$2@$k" "$1" || return 1; done; }

# Appraisal counts so far, from this run's log: "E1: 45 appraisal, 120 story"
kept() { grep -aE "^E[0-9]: [0-9]+ appraisal" "$LOG" 2>/dev/null | awk -v m="$MIN" '
  { n=$2+0; s=s sep $1 " " n (n < m ? " (too few)" : ""); sep=", " } END { print s }' | tr -d ':' ; }
event_now() { grep -aoE "^E[0-9]( comprehension|:)" "$LOG" 2>/dev/null | tail -n 1 | cut -c2; }

s1=0; s2=0; s3=0
has r2_data/rearing/train.jsonl && has r2_data/rearing_herb/train.jsonl && s1=1
has "$A/rearing/DONE" && has "$A/rearing_herb/DONE" && s2=1
scored base - && scored rearing - && scored rearing_herb - && s3=1

now=""; total_sets=0; done_sets=0; app_done=0; nseeds=0
for s in $SEEDS; do
  nseeds=$((nseeds + 1))
  has "r2_data/$MODEL/s$s/balance.json" && app_done=$((app_done + 1))
  for set in $SETS $STEP9; do
    total_sets=$((total_sets + 1))
    if set_done "$s" "$set"; then done_sets=$((done_sets + 1))
    elif [ -z "$cur_set" ] && has "r2_data/$MODEL/s$s/balance.json"; then
      cur_set="$set"; cur_seed="$s"; cur_n=$((done_sets + 1))
      if ! ready "$s" "$set"; then cur_what="appraising"
      elif has "$A/s$s/$set/DONE"; then cur_what="scoring"; else cur_what="training"; fi
    fi
  done
done
s4=$([ "$app_done" = "$nseeds" ] && echo 1 || echo 0)
s5=$([ "$done_sets" = "$total_sets" ] && echo 1 || echo 0)
s6=0; [ "$s5" = 1 ] && has "$R/summary.txt" && s6=1

if   [ $s1 = 0 ]; then now="Step 1 of 6: making the rearing data (Homer's and Herb's backgrounds)."
elif [ $s2 = 0 ]; then now="Step 2 of 6: training the rearing into the model."
elif [ $s3 = 0 ]; then now="Step 3 of 6: measuring the untrained and the reared model."
elif [ $s4 = 0 ]; then e=$(event_now)
  now="Step 4 of 6: Homer is writing his appraisals$([ -n "$e" ] && echo ", event $e of 5")."
elif [ $s5 = 0 ] && [ "$cur_what" = appraising ]; then
  now="Step 5 of 6: appraising for $cur_set$([ "$nseeds" -gt 1 ] && echo ", repeat $cur_seed")."
elif [ $s5 = 0 ]; then
  now="Step 5 of 6: $cur_what set $cur_n of $total_sets ($cur_set$([ "$nseeds" -gt 1 ] && echo ", repeat $cur_seed"))."
elif [ $s6 = 0 ]; then now="Step 6 of 6: writing the summary."
else now="All 6 steps done. Results are in $R."
fi
echo "$now"
echo
echo "  $(box [ $s1 = 1 ]) 1  rearing data"
echo "  $(box [ $s2 = 1 ]) 2  rearing training"
echo "  $(box [ $s3 = 1 ]) 3  measuring the untrained and reared model"
k=$(kept)
echo "  $(box [ $s4 = 1 ]) 4  Homer's appraisals$([ -n "$k" ] && echo "   kept so far: $k  (each needs $MIN)")"
echo "  $(box [ $s5 = 1 ]) 5  training and scoring the sets, then repair, Herb and claim 5   $done_sets of $total_sets done"
echo "  $(box [ $s6 = 1 ]) 6  summary"

if [ "$1" = "--details" ]; then
  for s in $SEEDS; do
    echo; echo "  Sets, repeat $s (scored at $(seq $EVERY $EVERY $ITERS | paste -sd/ -) steps):"
    for set in $SETS $STEP9; do
      line="    $(printf %-13s "$set")"
      for ((k=EVERY; k<=ITERS; k+=EVERY)); do scored "$set@$k" "$s" && line="$line  @$k ok" || line="$line  @$k --"; done
      echo "$line"
    done
  done
fi
