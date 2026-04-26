#!/bin/bash
# Collect grip_4 episodes for can / milk.
# Each episode is saved to its own file. Failed episodes are retried.
# Usage: bash run_collect_grip4_batch.sh [episodes_per_target]

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/zhx/miniconda3/envs/openpi311/bin/python}"
OUTDIR="dataset_auto_grip_4"
mkdir -p "$OUTDIR"

EPISODES_PER_TARGET="${1:-50}"
TARGETS=("can" "milk")

total_targets=${#TARGETS[@]}
total_episodes=$(( total_targets * EPISODES_PER_TARGET ))
global_idx=0
success_count=0

for TARGET in "${TARGETS[@]}"; do
    existing_max=$(find "$OUTDIR" -maxdepth 1 -type f -name "grip4_${TARGET}_*.hdf5" \
        | sed -E "s#.*_([0-9]{3})\.hdf5#\1#" \
        | sort -n \
        | tail -n 1)

    if [ -z "$existing_max" ]; then
        target_idx=0
    else
        target_idx=$existing_max
    fi

    for ep in $(seq 1 "$EPISODES_PER_TARGET"); do
        global_idx=$(( global_idx + 1 ))
        target_idx=$(( target_idx + 1 ))
        IDX=$(printf "%03d" "$target_idx")
        OUTFILE="${OUTDIR}/grip4_${TARGET}_${IDX}.hdf5"

        attempt=0
        success=0
        while [ "$success" -eq 0 ]; do
            attempt=$(( attempt + 1 ))
            echo "[$global_idx/$total_episodes] target=$TARGET ep=$target_idx (attempt=$attempt) -> $OUTFILE"

            "$PYTHON_BIN" collect_hannes_autocruise_grip_4.py \
                --episodes 1 \
                --target-object "$TARGET" \
                --output "$OUTFILE" \
                --output-format episode

            result=$("$PYTHON_BIN" - "$OUTFILE" <<'PY'
import sys, h5py
with h5py.File(sys.argv[1], 'r') as f:
    print(int(f.attrs.get('num_success', 0)))
PY
)

            if [ "$result" = "1" ]; then
                success=1
                success_count=$(( success_count + 1 ))
                echo "  -> OK"
            else
                echo "  -> FAILED (success=0), retrying..."
                rm -f "$OUTFILE"
            fi
        done
    done
done

echo ""
echo "=== All done: $success_count/$total_episodes succeeded. Output: $OUTDIR ==="