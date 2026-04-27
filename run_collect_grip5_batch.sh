#!/bin/bash
# Collect grip_5 episodes: target=milk/can, near=all others except target
# Each episode is saved to its own file. Failed episodes are retried.
# Usage: bash run_collect_grip5_batch.sh [episodes_per_combo]

OUTDIR="${OUTDIR:-dataset_auto_grip_5_rerun}"
mkdir -p "$OUTDIR"

EPISODES_PER_COMBO="${1:-5}"

# (target, near) pairs
COMBOS=(
    "milk can"
    "milk lemon"
    "milk bread"
    "milk hammer"
    "can milk"
    "can lemon"
    "can bread"
    "can hammer"
)

total_combos=${#COMBOS[@]}
total_episodes=$(( total_combos * EPISODES_PER_COMBO ))
global_idx=0
success_count=0

for combo in "${COMBOS[@]}"; do
    TARGET=$(echo "$combo" | cut -d' ' -f1)
    NEAR=$(echo "$combo" | cut -d' ' -f2)

    # Continue index from existing files to avoid overwriting previous runs.
    existing_max=$(ls "$OUTDIR"/grip5_${TARGET}_near_${NEAR}_*.hdf5 2>/dev/null \
        | sed -E "s#.*_([0-9]{3})\.hdf5#\1#" \
        | sort -n \
        | tail -n 1)
    if [ -z "$existing_max" ]; then
        combo_idx=0
    else
        combo_idx=$existing_max
    fi

    for ep in $(seq 1 "$EPISODES_PER_COMBO"); do
        global_idx=$(( global_idx + 1 ))
        combo_idx=$(( combo_idx + 1 ))
        IDX=$(printf "%03d" "$combo_idx")
        OUTFILE="${OUTDIR}/grip5_${TARGET}_near_${NEAR}_${IDX}.hdf5"

        attempt=0
        success=0
        while [ "$success" -eq 0 ]; do
            attempt=$(( attempt + 1 ))
            echo "[$global_idx/$total_episodes] target=$TARGET near=$NEAR ep=$combo_idx (attempt=$attempt) -> $OUTFILE"
            python collect_hannes_autocruise_grip_5.py \
                --episodes 1 \
                --target-object "$TARGET" \
                --place-near-object "$NEAR" \
                --output "$OUTFILE" \
                --output-format episode
            # Check success by reading attribute from the hdf5
            result=$(python - "$OUTFILE" <<'PY'
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
