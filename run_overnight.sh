#!/bin/bash
# Stage B overnight run on Adlerbot. Resumable: rerun to continue where it stopped.
# Override the models with: MODELS="org/model-a org/model-b" ./run_overnight.sh
cd "$(dirname "$0")"
source ~/venvs/bmem/bin/activate
MODELS="${MODELS:-mlx-community/Qwen3.5-4B-4bit mlx-community/Qwen3.8-27B-8bit}"
LAYERS="${LAYERS:-0.25,0.35,0.5,0.65,0.75}"
ALPHAS="${ALPHAS:-0.03,0.06,0.12,0.25,0.5,1.0}"
for m in $MODELS; do
  tag="${m##*/}"
  echo "=== $m  $(date)"
  python llm_test.py --model "$m" --layers "$LAYERS" --alphas "$ALPHAS" || echo "!!! $m stopped with an error"
  if [ -f "results/$tag/scores.jsonl" ]; then
    python analyze_llm.py "results/$tag" > "results/$tag/summary.txt" && echo "summary: results/$tag/summary.txt"
  fi
done
echo "=== B finished  $(date)"
[ "${SKIP_C:-0}" = 1 ] || ./run_c.sh
