#!/bin/bash
# Stage C, unattended, on Adlerbot: generate the training data from the local
# :8080 server, swap the server to a small model to free memory, wait for
# the stage B run to finish, then train and score the adapters.
cd "$(dirname "$0")"
source ~/venvs/bmem/bin/activate
URL="${URL:-http://127.0.0.1:8080/v1}"
echo "=== data  $(date)"
[ -f c_data/neutral/train.jsonl ] || python gen_c_data.py --url "$URL" || { echo "!!! data generation failed"; exit 1; }
echo "=== freeing server memory  $(date)"
curl -s -m 300 "$URL/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3.5-4B-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' > /dev/null
echo "=== waiting for stage B  $(date)"
while pgrep -f "[r]un_overnight.sh" > /dev/null; do sleep 60; done
echo "=== stage C  $(date)"
./run_c.sh
