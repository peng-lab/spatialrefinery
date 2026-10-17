#!/bin/bash

# Segment nuclei (InstanSeg) in every slide of a directory and write one SpatialData zarr per slide. One array task =
# one slide = one Slurm job on one GPU. Run this file directly (not with sbatch): it sizes the array to the number of
# slides and keeps at most MAX_CONCURRENT tasks running; Slurm starts the next pending task as soon as one ends:
#
#   cd <repo root> && slurm_scripts/segment_wsi_array.sbatch <wsi_dir> <zarr_outdir> <template_adata>
#
# Run outside Slurm it is the submitter; run by Slurm as an array task it is the worker. Task i processes the i-th slide
# of the sorted slide list. Re-submitting is safe; finished slides are skipped by the python scripts.
#
# Overrides (environment variables):
#   SEG_OUTDIR: where cells.geojson is written, default <zarr_outdir>/../segmentation
#   WSI_MPP: microns per pixel; set it for slides whose metadata is missing or < 0.1 um/px
#   TILE_SIZE, OVERLAP, CLAHE (e.g. 2.0), SEED_THRESHOLD (e.g. 0.4): InstanSeg settings, unset = script default
#   MAX_CONCURRENT: number of slides processed at once, default 4
#   PY

# parameters
#SBATCH --account=hai_1240
#SBATCH --job-name=instanseg-segment
#SBATCH --output=slurm-logs/instanseg-segment/%x-%A_%a.out
#SBATCH --error=slurm-logs/instanseg-segment/%x-%A_%a.err
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=500GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append

set -euo pipefail

# --- config -------------------------------------------------------------------------------------------------------
USAGE="usage: slurm_scripts/segment_wsi_array.sbatch <wsi_dir> <zarr_outdir> <template_adata>"
WSI_DIR="${1:?$USAGE}"
ZARR_OUTDIR="${2:?$USAGE}"
TEMPLATE_ADATA="${3:?$USAGE}"
SEG_OUTDIR="${SEG_OUTDIR:-$(dirname "$ZARR_OUTDIR")/segmentation}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
PY="${PY:-/p/project1/hai_1240/spatialrefinery/.venv/bin/python}"

# Every submitter and worker builds the same sorted list, so no manifest has to be shared. ._* are AppleDouble shadows.
list_slides() {
    find "$WSI_DIR" -maxdepth 1 -type f ! -name '._*' \
        \( -name '*.svs' -o -name '*.ndpi' -o -name '*.tif' -o -name '*.tiff' -o -name '*.mrxs' \) | sort
}

# --- submit -------------------------------------------------------------------------------------------------------
# Not inside a Slurm job: size the array to the slide count and resubmit this file as the worker.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ -d "$WSI_DIR" ]] || { echo "ERROR: no such directory: $WSI_DIR" >&2; exit 2; }
    [[ -f "$TEMPLATE_ADATA" ]] || { echo "ERROR: template AnnData not found: $TEMPLATE_ADATA" >&2; exit 2; }
    N_SLIDES=$(list_slides | wc -l)
    [[ "$N_SLIDES" -gt 0 ]] || { echo "ERROR: no slides found in $WSI_DIR" >&2; exit 1; }
    mkdir -p slurm-logs/instanseg-segment
    exec sbatch --array="0-$((N_SLIDES - 1))%${MAX_CONCURRENT}" "${BASH_SOURCE[0]}" "$WSI_DIR" "$ZARR_OUTDIR" "$TEMPLATE_ADATA"
fi

# --- worker (one array task) --------------------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mapfile -t SLIDES < <(list_slides)
WSI="${SLIDES[${SLURM_ARRAY_TASK_ID:?plain sbatch makes a single task; run the file directly so it submits an array}]:?array task beyond the ${#SLIDES[@]} slides in $WSI_DIR}"
GEOJSON="$SEG_OUTDIR/$(basename "$WSI")/cells.geojson"

OPTIONAL_ARGS=()
[[ -n "${WSI_MPP:-}" ]] && OPTIONAL_ARGS+=(--wsi-mpp "$WSI_MPP")
[[ -n "${TILE_SIZE:-}" ]] && OPTIONAL_ARGS+=(--tile-size "$TILE_SIZE")
[[ -n "${OVERLAP:-}" ]] && OPTIONAL_ARGS+=(--overlap "$OVERLAP")
[[ -n "${CLAHE:-}" ]] && OPTIONAL_ARGS+=(--clahe "$CLAHE")
[[ -n "${SEED_THRESHOLD:-}" ]] && OPTIONAL_ARGS+=(--seed-threshold "$SEED_THRESHOLD")

export PYTHONWARNINGS=ignore
echo "slide $((SLURM_ARRAY_TASK_ID + 1))/${#SLIDES[@]}: $WSI on $(hostname), started $(date -Is)"

# Stage 1 and 2 share a cells.geojson so a failed conversion does not force re-segmentation.
srun "$PY" scripts/instanseg_segment.py --wsi-path "$WSI" --outdir "$SEG_OUTDIR" "${OPTIONAL_ARGS[@]}"
srun "$PY" scripts/geojson_to_spatialdata.py --geojson-path "$GEOJSON" --zarr-outdir "$ZARR_OUTDIR" \
    --wsi-path "$WSI" --template-adata "$TEMPLATE_ADATA"

echo "finished $(date -Is)"
