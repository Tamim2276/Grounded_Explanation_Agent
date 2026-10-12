#!/usr/bin/env bash
# Day 7.4: one command for after every power cut (Task Scheduler runs it at log-on: 7.5).
# Starts the model server when there is room, waits until it answers, then continues the
# three systems on the test set -- each run skips the questions already in its file. When
# all three are complete: the second pass (each photo claim boxed by its own map), then a
# checkpoint. Running it again after that changes nothing.
#
#   bash scripts/resume_eval.sh              # watch it: tail -f data/logs/resume_eval.log
#   SPLIT=dev bash scripts/resume_eval.sh    # the same on dev (a rehearsal)
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/logs
exec >> data/logs/resume_eval.log 2>&1          # unattended: everything from here goes to the log
echo "=== $(date '+%F %T'): power is back, resuming"
source .venv/Scripts/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
SPLIT="${SPLIT:-test}"
SYSTEMS="full single_pass text_only"

server_up() { curl -sf -o /dev/null http://localhost:8080/v1/models; }
server_running() { tasklist //FI "IMAGENAME eq llama-server.exe" | grep -q llama-server; }

# 1. ColQwen2's question vectors, before the model server takes the GPU (instant when cached).
python scripts/cache_question_vectors.py --split "$SPLIT" || { echo "question vectors missing; stopped"; exit 1; }

# 2. The model server: started when there is room (start_llm.sh checks memory and GPU, and
#    refuses while another job needs them -- then this tries again every minute).
until server_up; do
    server_running || bash scripts/start_llm.sh >> data/logs/llm.log 2>&1 &
    sleep 60
done
echo "$(date '+%T') model server answers"

# 3. The three systems, in order. A crash of the PC side (out of memory) is retried.
for s in $SYSTEMS; do
    for attempt in 1 2 3; do
        python scripts/run_eval.py --system "$s" --split "$SPLIT" && break
        echo "$(date '+%T') run_eval $s stopped (attempt $attempt); retrying in 2 min"
        sleep 120
    done
    bash scripts/checkpoint.sh "evaluation: $s on $SPLIT" || true     # no internet: saved next time
done

# 4. The model server is no longer needed: free the GPU and the memory.
taskkill //IM llama-server.exe //F > /dev/null 2>&1

# 5. The stronger-model comparison: the full system with Gemini (needs internet and the key in
#    .env; without internet the runner waits). Skipped when there is no key.
if python -c "import gea.llm as l; l.gemini_key()" 2>/dev/null; then
    python scripts/run_eval.py --system full --split "$SPLIT" --provider gemini
    bash scripts/checkpoint.sh "evaluation: full with Gemini on $SPLIT" || true
fi

# 6. The second pass: cache ColQwen2 vectors of the claims about photos, then replay those
#    questions from the cache with the claims' own maps.
files=""
for s in $SYSTEMS; do files="$files results/runs/${s}_${SPLIT}.jsonl"; done
[ -f "results/runs/full_gemini_${SPLIT}.jsonl" ] && files="$files results/runs/full_gemini_${SPLIT}.jsonl"
if python scripts/claim_vectors.py $files; then
    for s in $SYSTEMS; do python scripts/run_eval.py --system "$s" --split "$SPLIT" --rebox; done
    [ -f "results/runs/full_gemini_${SPLIT}.jsonl" ] && python scripts/run_eval.py --system full --split "$SPLIT" --provider gemini --rebox
fi
bash scripts/checkpoint.sh "evaluation complete on $SPLIT" || true
echo "=== $(date '+%F %T'): all done"
