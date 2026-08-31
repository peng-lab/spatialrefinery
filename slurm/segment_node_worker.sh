#!/bin/bash

# ==============================================================================
# InstanSeg + SpatialData: per-node worker
# ==============================================================================
# Called by srun on each allocated node. Reads a manifest chunk (one WSI path
# per line) and processes the slides in parallel across the node's GPUs, using
# round-robin assignment.
#
# Both stages run in the same interpreter, from the project venv.
#
# They exchange a cells.geojson rather than being merged: it is a resume
# boundary, so a conversion failure does not force re-segmentation.
#
# Usage: invoked by slurm/segment_slurm.sh, not directly.
#   bash segment_node_worker.sh <manifest_chunk> <seg_outdir> <zarr_outdir> \
#        <template_adata> <gpus_per_node> [wsi_mpp] [tile_size] [overlap] \
#        [clahe_clip] [seed_threshold]
# ==============================================================================

set -euo pipefail

MANIFEST_CHUNK="$1"
SEG_OUTDIR="$2"
ZARR_OUTDIR="$3"
TEMPLATE_ADATA="$4"
GPUS_PER_NODE="${5:-4}"
WSI_MPP="${6:-}"
TILE_SIZE="${7:-512}"
OVERLAP="${8:-80}"
CLAHE="${9:-}"           # empty = off
SEED_THRESHOLD="${10:-}" # empty = model default (0.7)

REPO_ROOT="${REPO_ROOT:-/p/project1/hai_1240/spatialrefinery}"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"

SEGMENT_SCRIPT="${REPO_ROOT}/scripts/instanseg_segment.py"
CONVERT_SCRIPT="${REPO_ROOT}/scripts/geojson_to_spatialdata.py"

LOG_DIR="${LOG_DIR:-$(dirname "$ZARR_OUTDIR")/slurm_logs}"
mkdir -p "$LOG_DIR"

if [ ! -x "$PY" ]; then
    echo "ERROR: interpreter not found: $PY" >&2
    echo "   Create it with: uv sync --extra segmentation" >&2
    exit 2
fi

echo "======================================================="
echo "NODE WORKER: $(hostname)"
echo "  Manifest chunk: $MANIFEST_CHUNK"
echo "  GPUs per node:  $GPUS_PER_NODE"
echo "  Interpreter:    $PY"
echo "  Started at:     $(date -Is)"
echo "======================================================="

mapfile -t WSI_LIST < "$MANIFEST_CHUNK"
TOTAL_WSIS=${#WSI_LIST[@]}

if [ "$TOTAL_WSIS" -eq 0 ]; then
    echo "No WSIs in manifest chunk. Nothing to do."
    exit 0
fi

echo "Processing $TOTAL_WSIS WSIs across $GPUS_PER_NODE GPUs..."

OPTIONAL_ARGS=()
if [ -n "$WSI_MPP" ]; then
    OPTIONAL_ARGS+=(--wsi-mpp "$WSI_MPP")
fi
if [ -n "$CLAHE" ]; then
    OPTIONAL_ARGS+=(--clahe "$CLAHE")
fi
if [ -n "$SEED_THRESHOLD" ]; then
    OPTIONAL_ARGS+=(--seed-threshold "$SEED_THRESHOLD")
fi

process_wsi() {
    local WSI_PATH="$1"
    local GPU_ID="$2"
    local SAMPLE_NAME
    SAMPLE_NAME=$(basename "$WSI_PATH")
    local LOG_FILE="${LOG_DIR}/${SAMPLE_NAME}_gpu${GPU_ID}.log"

    {
        echo "### [$SAMPLE_NAME] GPU $GPU_ID -- $(date -Is)"

        echo "--- Stage 1: InstanSeg segmentation ---"
        # The process sees one GPU, so --gpu-id stays 0.
        local SEGMENT_OUT
        SEGMENT_OUT=$(CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$SEGMENT_SCRIPT" \
            --wsi-path "$WSI_PATH" \
            --outdir "$SEG_OUTDIR" \
            --gpu-id 0 \
            --tile-size "$TILE_SIZE" \
            --overlap "$OVERLAP" \
            "${OPTIONAL_ARGS[@]}" 2>&1)
        echo "$SEGMENT_OUT"

        local GEOJSON_PATH
        GEOJSON_PATH=$(echo "$SEGMENT_OUT" | grep '^GEOJSON_PATH=' | tail -1 | cut -d= -f2-)
        if [ -z "$GEOJSON_PATH" ]; then
            GEOJSON_PATH="$SEG_OUTDIR/$SAMPLE_NAME/cells.geojson"
        fi
        if [ ! -f "$GEOJSON_PATH" ]; then
            echo "FAILED: stage 1 produced no cells.geojson at $GEOJSON_PATH"
            exit 1
        fi
        echo "Stage 1 complete: $GEOJSON_PATH"

        echo "--- Stage 2: SpatialData conversion ---"
        "$PY" "$CONVERT_SCRIPT" \
            --geojson-path "$GEOJSON_PATH" \
            --zarr-outdir "$ZARR_OUTDIR" \
            --wsi-path "$WSI_PATH" \
            --template-adata "$TEMPLATE_ADATA"

        echo "[$SAMPLE_NAME] done at $(date -Is)"
    } > "$LOG_FILE" 2>&1
}

for i in "${!WSI_LIST[@]}"; do
    WSI_PATH="${WSI_LIST[$i]}"
    GPU_ID=$((i % GPUS_PER_NODE))
    echo "---> [$((i + 1))/$TOTAL_WSIS] $(basename "$WSI_PATH") -> GPU $GPU_ID"

    process_wsi "$WSI_PATH" "$GPU_ID" &

    if (( $(jobs -rp | wc -l) >= GPUS_PER_NODE )); then
        wait -n || echo "A worker exited non-zero (continuing with remaining WSIs)"
    fi
done

echo "Waiting for remaining workers..."
wait

echo "======================================================="
echo "NODE WORKER: $(hostname) - COMPLETE"
echo "  Processed:   $TOTAL_WSIS WSIs"
echo "  Finished at: $(date -Is)"
echo "======================================================="
