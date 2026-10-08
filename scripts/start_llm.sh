#!/usr/bin/env bash
# Start the local language model on the B580 (BUILD_PLAN.md, Days 1 and 5).
# Run it again after every power cut. It needs no internet: the model files were
# downloaded once by scripts/download_all.py.
#
#   bash scripts/start_llm.sh           # 8192 tokens of context
#   bash scripts/start_llm.sh 6144      # smaller, if the GPU runs out of memory
cd "$(dirname "$0")/.." || exit 1
context="${1:-8192}"
port="${2:-8080}"

model="data/models/qwen2.5-vl-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
mmproj="data/models/qwen2.5-vl-7b/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
for f in "$model" "$mmproj"; do
    [ -f "$f" ] || { echo "Missing $f -- run: python scripts/download_all.py"; exit 1; }
done

# llama-server from PATH (winget install), otherwise the unzipped release folder
server="$(command -v llama-server)"
[ -n "$server" ] || server="/d/tools/llama.cpp/llama-server.exe"
[ -x "$server" ] || { echo "llama-server not found -- BUILD_PLAN.md, Day 1, step 1.7"; exit 1; }

# The model needs about 7.5 GB of the B580's 12 GB. If another program is already using
# the GPU (another experiment, a game), llama.cpp crashes with "ErrorOutOfDeviceMemory",
# so ask Windows first and name the programs. (llama.cpp's own "MiB free" figure does not
# count other programs.) FORCE=1 skips the check.
if [ -z "$FORCE" ]; then
    used_mb=$(powershell.exe -NoProfile -Command '[int](((Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage").CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1MB)' 2>/dev/null | tr -d '\r')
    if [ -n "$used_mb" ] && [ "$used_mb" -gt 4500 ]; then
        echo "The GPU already has $used_mb MB in use; the model needs about 7500 MB of the 12 GB."
        echo "Programs using the GPU now:"
        powershell.exe -NoProfile -Command '(Get-Counter "\GPU Process Memory(*)\Dedicated Usage").CounterSamples | Where-Object { $_.CookedValue -gt 200MB } | Sort-Object CookedValue -Descending | ForEach-Object { $id = [regex]::Match($_.InstanceName, "pid_(\d+)").Groups[1].Value; "  {0,6:N0} MB  {1} (pid {2})" -f ($_.CookedValue / 1MB), (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName, $id }' 2>/dev/null | tr -d '\r'
        echo "Stop them or wait until they finish, then run this again."
        exit 1
    fi
fi

# -ngl 99: every layer on the GPU   -c: context tokens   -np 1: one request at a time
exec "$server" -m "$model" --mmproj "$mmproj" -ngl 99 -c "$context" -np 1 --port "$port"
