#!/bin/bash

# ==============================================================================
# SLURM launcher: InstanSeg segmentation + SpatialData conversion
# ==============================================================================
# Discovers WSI files, writes a manifest, and submits a job that spreads the
# slides across nodes (each with several GPUs).
#
# Usage:
#   ./segment_slurm.sh <wsi_dir> <seg_outdir> <zarr_outdir> <template_adata> [sample_subset]
#
# Examples:
#   ./slurm/segment_slurm.sh /data/wsis /data/seg /data/zarr /data/template.h5ad
#   NODES=4 ./slurm/segment_slurm.sh /data/wsis /data/seg /data/zarr /data/template.h5ad s1,s2
# ==============================================================================

set -euo pipefail

ACCOUNT="${ACCOUNT:-hai_1240}"
PARTITION="${PARTITION:-dc-hwai}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TIME="${TIME:-24:00:00}"
MEMORY="${MEMORY:-500G}"
CPUS_PER_NODE="${CPUS_PER_NODE:-64}"

# Segmentation settings. InstanSeg tiles internally, so there is no batch size
# or worker pool to size here.
WSI_MPP="${WSI_MPP:-}"       # empty = read from slide metadata
TILE_SIZE="${TILE_SIZE:-512}"
OVERLAP="${OVERLAP:-80}"
# Sensitivity, both off by default. Worth setting on weakly stained cohorts:
# CLAHE=2.0 with SEED_THRESHOLD=0.4 recovered 27% more nuclei on a pale kidney H&E.
CLAHE="${CLAHE:-}"
SEED_THRESHOLD="${SEED_THRESHOLD:-}"

WSI_EXTENSIONS=("svs" "ndpi" "ome.tif" "ome.tiff" "tif" "tiff" "mrxs")

REPO_ROOT="${REPO_ROOT:-/p/project1/hai_1240/spatialrefinery}"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"
NODE_WORKER_SCRIPT="${REPO_ROOT}/slurm/segment_node_worker.sh"

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    cat <<USAGE
Usage: $0 <wsi_directory> <seg_outdir> <zarr_outdir> <template_adata> [comma_separated_sample_ids]

Environment variables:
  NODES=4           Number of nodes (default: 1)
  GPUS_PER_NODE=4   GPUs per node (default: 4)
  WSI_MPP=0.2738    Microns per pixel (default: read from slide metadata)
  TILE_SIZE=512     InstanSeg tile size (default: 512)
  OVERLAP=80        Tile overlap in pixels (default: 80)
  CLAHE=2.0         CLAHE clip limit for pale slides (default: off)
  SEED_THRESHOLD=0.4  Model seed threshold (default: off, model uses 0.7)
  TIME=24:00:00     Max runtime
  MEMORY=500G       Memory per node
  REPO_ROOT=...     Repo checkout (default: $REPO_ROOT)
USAGE
    exit 1
fi

WSI_DIR="$1"
SEG_OUTDIR="$2"
ZARR_OUTDIR="$3"
TEMPLATE_ADATA="$4"
SAMPLE_SUBSET="${5:-}"

BASE_PATH="$(dirname "$ZARR_OUTDIR")"
LOG_DIR="${BASE_PATH}/slurm_logs"
MANIFEST_DIR="${BASE_PATH}/manifests"

mkdir -p "$SEG_OUTDIR" "$ZARR_OUTDIR" "$LOG_DIR" "$MANIFEST_DIR"

if [ ! -d "$WSI_DIR" ]; then
    echo "ERROR: WSI directory not found: $WSI_DIR" >&2
    exit 1
fi
if [ ! -f "$TEMPLATE_ADATA" ]; then
    echo "ERROR: template AnnData not found: $TEMPLATE_ADATA" >&2
    exit 1
fi
if [ ! -f "$NODE_WORKER_SCRIPT" ]; then
    echo "ERROR: node worker not found: $NODE_WORKER_SCRIPT" >&2
    exit 1
fi
if [ ! -x "$PY" ]; then
    echo "ERROR: interpreter not found: $PY" >&2
    echo "   Create it with: uv sync --extra segmentation" >&2
    exit 1
fi

echo "Scanning for WSI files in: $WSI_DIR"
MANIFEST="${MANIFEST_DIR}/manifest_$(date +%Y%m%d_%H%M%S).txt"
: > "$MANIFEST"

if [ -n "$SAMPLE_SUBSET" ]; then
    echo "  Filtering to: $SAMPLE_SUBSET"
    IFS=',' read -r -a SAMPLE_IDS <<< "$SAMPLE_SUBSET"
    for sample_id in "${SAMPLE_IDS[@]}"; do
        for ext in "${WSI_EXTENSIONS[@]}"; do
            [ -f "$WSI_DIR/${sample_id}.${ext}" ] && echo "$WSI_DIR/${sample_id}.${ext}" >> "$MANIFEST"
        done
    done
else
    for ext in "${WSI_EXTENSIONS[@]}"; do
        find "$WSI_DIR" -maxdepth 1 -name "*.${ext}" -type f >> "$MANIFEST" 2>/dev/null || true
    done
fi

sort -u -o "$MANIFEST" "$MANIFEST"
TOTAL_WSIS=$(wc -l < "$MANIFEST")

if [ "$TOTAL_WSIS" -eq 0 ]; then
    echo "No WSI files found. Exiting."
    rm -f "$MANIFEST"
    exit 0
fi

cat <<SUMMARY

Found $TOTAL_WSIS WSIs.
Manifest: $MANIFEST

  Nodes:       $NODES
  GPUs/node:   $GPUS_PER_NODE
  Total GPUs:  $((NODES * GPUS_PER_NODE))
  Tile size:   $TILE_SIZE (overlap $OVERLAP)
  CLAHE:       ${CLAHE:-off}
  Seed thr.:   ${SEED_THRESHOLD:-model default}
  WSI MPP:     ${WSI_MPP:-from slide metadata}

SUMMARY

JOB_NAME="instanseg_sdata_${TOTAL_WSIS}wsis"
echo "Submitting SLURM job: $JOB_NAME"

sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_NODE}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/instanseg_job_%j.txt
#SBATCH --error=${LOG_DIR}/instanseg_job_%j.err.txt

echo "======================================================="
echo "SLURM JOB: InstanSeg + SpatialData"
echo "Job ID:    \$SLURM_JOB_ID"
echo "Nodes:     \$SLURM_JOB_NODELIST"
echo "Started:   \$(date -Is)"
echo "======================================================="

export REPO_ROOT="${REPO_ROOT}"
export LOG_DIR="${LOG_DIR}"
export PYTHONWARNINGS=ignore

CHUNK_PREFIX="${MANIFEST_DIR}/chunk_\${SLURM_JOB_ID}_"
split -n l/\${SLURM_NNODES} "${MANIFEST}" "\${CHUNK_PREFIX}"

CHUNK_FILES=(\${CHUNK_PREFIX}*)
echo "Split manifest into \${#CHUNK_FILES[@]} chunks for \${SLURM_NNODES} nodes"

mapfile -t NODE_ARRAY < <(scontrol show hostnames \$SLURM_JOB_NODELIST)

for i in "\${!CHUNK_FILES[@]}"; do
    CHUNK_FILE="\${CHUNK_FILES[\$i]}"
    TARGET_NODE="\${NODE_ARRAY[\$i]}"
    echo "---> Node \$i (\$TARGET_NODE): \$(wc -l < "\$CHUNK_FILE") WSIs"

    srun --nodes=1 --ntasks=1 --exclusive \\
        --nodelist="\$TARGET_NODE" \\
        bash ${NODE_WORKER_SCRIPT} \\
            "\$CHUNK_FILE" \\
            "${SEG_OUTDIR}" \\
            "${ZARR_OUTDIR}" \\
            "${TEMPLATE_ADATA}" \\
            "${GPUS_PER_NODE}" \\
            "${WSI_MPP}" \\
            "${TILE_SIZE}" \\
            "${OVERLAP}" \\
            "${CLAHE}" \\
            "${SEED_THRESHOLD}" &
done

wait

echo "======================================================="
echo "JOB COMPLETE  \$(date -Is)"
echo "======================================================="

rm -f \${CHUNK_PREFIX}*
EOF

echo
echo "Submitted. Monitor with: squeue -u \$USER"
echo "  Logs:     $LOG_DIR"
echo "  Manifest: $MANIFEST"
