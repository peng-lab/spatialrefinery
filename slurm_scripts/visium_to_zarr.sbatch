#!/bin/bash

# Convert every Visium / CytAssist bundle of a panel to a SpatialData zarr. One array task = one bundle, at most
# MAX_CONCURRENT tasks at a time. Run this file directly (not with sbatch): it sizes the array to the number of bundles
# and resubmits itself as the worker.
#
#   cd <repo root> && slurm_scripts/visium_to_zarr.sbatch [panel/cohort]     # default: cytassist_WT/TENX
#
# Per bundle:
#   1. extract spatial/ from *_spatial.tar.gz if it is not there yet,
#   2. segment nuclei in <sample>_tissue_image.ome.tif with InstanSeg, unless its cells.geojson already exists,
#   3. convert the whole bundle, plus the nuclei, to <sample>.zarr and <sample>.zarr.zip.
# The input layout <panel>/<cohort>/<sample> is mirrored under SEG_ROOT and ZARR_ROOT. Finished samples are skipped, so
# resubmitting after a failure only redoes what is missing. Bundles without a filtered counts matrix are not listed.

# parameters
#SBATCH --account=hai_1240
#SBATCH --job-name=visium-convert
#SBATCH --output=slurm-logs/visium-convert/%x-%A_%a.out
#SBATCH --error=slurm-logs/visium-convert/%x-%A_%a.err
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
PANEL="${1:-cytassist_WT/TENX}" # input directory relative to RAW_ROOT
MAX_CONCURRENT=4
RAW_ROOT=/p/project1/hai_1240/datasets/orthogonal/raw_output_bundle
SEG_ROOT=/p/scratch/hai_1240/datasets/orthogonal/segmentation_files
ZARR_ROOT=/p/scratch/hai_1240/datasets/orthogonal/spatialdata_zarr
TEMPLATE_ADATA=/p/project1/hai_1240/datasets/templates/multi_tissue.h5ad
PY=/p/project1/hai_1240/spatialrefinery/.venv/bin/python

# Every submitter and worker builds the same sorted list. A bundle with no counts matrix (e.g. Visium_V2_Human_Lymph_Node)
# cannot be converted, so it is left out here rather than failing a task.
list_slides() {
    for WSI in "$RAW_ROOT/$PANEL"/*/*_tissue_image.ome.tif; do
        if [[ -f "${WSI%_tissue_image.ome.tif}_filtered_feature_bc_matrix.h5" ]]; then echo "$WSI"; fi
    done
}

# --- submit -------------------------------------------------------------------------------------------------------
# Not inside a Slurm job: size the array to the bundle count and resubmit this file as the worker.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    N_SLIDES=$(list_slides | wc -l)
    [[ "$N_SLIDES" -gt 0 ]] || { echo "ERROR: no convertible bundles under $RAW_ROOT/$PANEL" >&2; exit 1; }
    mkdir -p slurm-logs/visium-convert
    exec sbatch --array="0-$((N_SLIDES - 1))%${MAX_CONCURRENT}" "${BASH_SOURCE[0]}" "$PANEL"
fi

# --- worker (one array task = one bundle) -------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
export PYTHONWARNINGS=ignore
# The shared venv has the main checkout installed editable; this makes it import this checkout instead.
export PYTHONPATH="$PWD/src"

mapfile -t SLIDES < <(list_slides)
WSI="${SLIDES[${SLURM_ARRAY_TASK_ID:?run the file directly so it submits an array}]:?array task beyond the ${#SLIDES[@]} bundles}"
SAMPLE=$(basename "$(dirname "$WSI")")
BUNDLE=$RAW_ROOT/$PANEL/$SAMPLE
SEG_DIR=$SEG_ROOT/$PANEL/$SAMPLE
GEOJSON="$SEG_DIR/$(basename "$WSI")/cells.geojson" # where segment_wsi writes it
echo "=== $SAMPLE ($((SLURM_ARRAY_TASK_ID + 1))/${#SLIDES[@]}) on $(hostname), started $(date -Is)"

# The converter reads spot positions from spatial/, which the bundle ships packed.
[[ -d "$BUNDLE/spatial" ]] || tar -xzf "$BUNDLE/${SAMPLE}_spatial.tar.gz" -C "$BUNDLE"

# Pixel size from the spot lattice. The raw TIFF tags are not trustworthy and InstanSeg needs microns per pixel.
MPP=$("$PY" -c "from spatialrefinery.io.visium import visium_he_mpp; print(visium_he_mpp('$BUNDLE/spatial/tissue_positions.csv'))")
echo "H&E pixel size: $MPP um/px"

if [[ -f "$GEOJSON" ]]; then
    echo "reusing $GEOJSON"
else
    "$PY" scripts/instanseg_segment.py --wsi-path "$WSI" --outdir "$SEG_DIR" --wsi-mpp "$MPP"
fi

"$PY" scripts/visium_to_zarr.py \
    --data-path "$RAW_ROOT/$PANEL" \
    --biospecimen-id "$SAMPLE" \
    --output-path "$ZARR_ROOT/$PANEL" \
    --nucleus-geojson "$GEOJSON" \
    --template-adata "$TEMPLATE_ADATA" \
    --keep-zarr

echo "finished $(date -Is)"
