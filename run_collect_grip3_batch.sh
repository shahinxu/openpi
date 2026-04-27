#!/bin/bash
# Collect 100 grip_3 episodes (50/50 can vs milk), one file each.
set -e

OUTDIR="dataset_auto_grip_3"
mkdir -p "$OUTDIR"

OBJECTS=("can" "milk")
TOTAL=100
CAN_IDX=0
MILK_IDX=0

for i in $(seq 1 $TOTAL); do
    OBJ=${OBJECTS[$((RANDOM % 2))]}
    if [ "$OBJ" = "can" ]; then
        CAN_IDX=$((CAN_IDX + 1))
        IDX=$(printf "%03d" $CAN_IDX)
        OUTFILE="${OUTDIR}/Grip the can_${IDX}.hdf5"
    else
        MILK_IDX=$((MILK_IDX + 1))
        IDX=$(printf "%03d" $MILK_IDX)
        OUTFILE="${OUTDIR}/Grip the milk_${IDX}.hdf5"
    fi
    echo "[$i/$TOTAL] target=$OBJ -> $OUTFILE"
    python collect_hannes_autocruise_grip_3.py \
        --episodes 1 \
        --target-object "$OBJ" \
        --output "$OUTFILE" \
        --output-format episode
done

echo "=== All $TOTAL episodes done. Output: $OUTDIR ==="
