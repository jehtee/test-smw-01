#!/bin/bash
# Stage C on Adlerbot: train LoRA adapters on c_data/, then score each with llm_test.py.
# Rearing is trained first; every perturbation is trained on top of it, with
# checkpoints every $EVERY iterations scored separately (the dose curve).
# Resumable: finished adapters and scored conditions are skipped.
cd "$(dirname "$0")"
source ~/venvs/bmem/bin/activate
MODELS="${MODELS:-mlx-community/Qwen3.5-4B-4bit mlx-community/Qwen3.8-27B-8bit}"
SETS="${SETS:-impl_friend fact_friend impl_stranger neutral}"
ITERS="${ITERS:-150}"; EVERY="${EVERY:-50}"; LR="${LR:-5e-5}"
MASK=""; mlx_lm.lora --help 2>/dev/null | grep -q -- --mask-prompt && MASK="--mask-prompt"

train() {  # model set out_dir [resume_file]
  [ -f "$3/adapters.safetensors" ] && { echo "have $3"; return 0; }
  mlx_lm.lora --model "$1" --train --data "c_data/$2" --adapter-path "$3" \
    --iters "$ITERS" --save-every "$EVERY" --batch-size 4 --learning-rate "$LR" \
    --steps-per-eval "$ITERS" $MASK ${4:+--resume-adapter-file "$4"}
}

for m in $MODELS; do
  tag="${m##*/}"; A="adapters/$tag"
  echo "=== C $m  $(date)"
  train "$m" rearing "$A/rearing" || { echo "!!! rearing failed for $m"; continue; }
  python llm_test.py --model "$m" --adapter "$A/rearing" --label rearing
  for s in $SETS; do
    train "$m" "$s" "$A/$s" "$A/rearing/adapters.safetensors" || { echo "!!! $s failed for $m"; continue; }
    for ck in "$A/$s"/0*_adapters.safetensors; do
      n=$((10#$(basename "$ck" | cut -d_ -f1)))
      d="$A/$s@$n"; mkdir -p "$d"
      cp "$A/$s/adapter_config.json" "$d/" && cp "$ck" "$d/adapters.safetensors"
      python llm_test.py --model "$m" --adapter "$d" --label "$s@$n"
    done
  done
  python analyze_llm.py "results/$tag" > "results/$tag/summary.txt" && echo "summary: results/$tag/summary.txt"
done
echo "=== C finished  $(date)"
