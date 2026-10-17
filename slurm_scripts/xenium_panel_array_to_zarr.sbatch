#!/bin/bash

# Convert one Xenium bundle of a panel to a zipped SpatialData zarr. One array task = one sample = one Slurm job on its
# own node, so every sample gets its own wall-clock limit. Run this file directly (not with sbatch): it sizes the array
# to the number of bundles and keeps at most MAX_CONCURRENT tasks running (dc-hwai lets one user hold 4 nodes); Slurm
# starts the next pending task as soon as a running one ends:
#
#   cd <repo root> && slurm_scripts/xenium_panel_array_to_zarr.sbatch <panel>
#
# Run outside Slurm it is the submitter; run by Slurm as an array task it is the worker. Task i converts the i-th bundle of the sorted bundle list, so no sample is ever run twice by one submission.
# Re-submitting is safe; finished samples are skipped by the python script.
#
# Overrides (environment variables):
#   EXTRA_ARGS="--overwrite" slurm_scripts/xenium_panel_array_to_zarr.sbatch <panel>   extra flags for the python script
#   RAW_ROOT, OUT_ROOT, NUM_WORKERS, PY
#   MAX_CONCURRENT: number of samples converted at once, default 4

# parameters
#SBATCH --account=hai_1240
#SBATCH --job-name=xenium-convert
#SBATCH --output=slurm-logs/xenium-convert/%x-%A_%a.out
#SBATCH --error=slurm-logs/xenium-convert/%x-%A_%a.err
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --open-mode=append
#SBATCH --mem=500GB
#SBATCH --time=24:00:00

set -euo pipefail

# --- config -------------------------------------------------------------------------------------------------------
PANEL="${1:?usage: slurm_scripts/xenium_panel_array_to_zarr.sbatch <panel>}"
RAW_ROOT="${RAW_ROOT:-/p/project1/hai_1240/datasets/flagship/raw_output_bundle}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
OUT_ROOT="${OUT_ROOT:-/p/scratch/hai_1240/sandbox/zarr_files}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PY="${PY:-/p/project1/hai_1240/spatialrefinery/.venv/bin/python}"
SCRIPT=scripts/xenium_panel_array_to_zarr.py
FAIL_LOG="slurm-logs/failed_samples.tsv"
EXIT_FAILURE_LOGGED=1 # exit code the python script returns after it has written its own failure row

# --- submit -------------------------------------------------------------------------------------------------------
# Not inside a Slurm job: size the array to the bundle count and resubmit this file as the worker.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ -d "$RAW_ROOT/$PANEL" ]] || { echo "ERROR: no such panel: $RAW_ROOT/$PANEL" >&2; exit 2; }
    N_SAMPLES=$(find "$RAW_ROOT/$PANEL" -mindepth 2 -maxdepth 4 -name experiment.xenium | wc -l)
    [[ "$N_SAMPLES" -gt 0 ]] || { echo "ERROR: no Xenium bundles found under $RAW_ROOT/$PANEL" >&2; exit 1; }
    mkdir -p slurm-logs/xenium-convert
    exec sbatch --array="0-$((N_SAMPLES - 1))%${MAX_CONCURRENT}" "${BASH_SOURCE[0]}" "$PANEL"
fi

# --- worker (one array task) --------------------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:?plain sbatch makes a single task; run the file directly so it submits an array}"
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"
[[ -f "$SCRIPT" ]] || { echo "ERROR: $SCRIPT not found under $REPO_ROOT; submit from the repo root" >&2; exit 2; }
[[ -d "$RAW_ROOT/$PANEL" ]] || { echo "ERROR: no such panel: $RAW_ROOT/$PANEL" >&2; exit 2; }

# No conda: the repo's uv venv already has spatialdata/spatialdata-io/openslide as wheels, and the system python3 is
# 3.9 (too old for this package).
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS="$NUM_WORKERS" OPENBLAS_NUM_THREADS="$NUM_WORKERS" MKL_NUM_THREADS="$NUM_WORKERS"

# --- sample list --------------------------------------------------------------------------------------------------
# Every worker builds the same sorted list, so no manifest has to be shared. -name is an exact match, so the
# AppleDouble shadows (._experiment.xenium) that live alongside some samples are never mistaken for a bundle root.
# Bundles sit at <panel>/<cohort>/<sample>/experiment.xenium.
mapfile -t SAMPLES < <(find "$RAW_ROOT/$PANEL" -mindepth 2 -maxdepth 4 -name experiment.xenium -printf '%h\n' | sort)
if [[ "${#SAMPLES[@]}" -eq 0 ]]; then
    echo "ERROR: no Xenium bundles found under $RAW_ROOT/$PANEL" >&2
    exit 1
fi
if [[ "$TASK_ID" -ge "${#SAMPLES[@]}" ]]; then
    echo "ERROR: array task $TASK_ID but only ${#SAMPLES[@]} bundles under $RAW_ROOT/$PANEL" >&2
    exit 1
fi
if [[ "$TASK_ID" -eq 0 ]]; then
    printf '%s\n' "${SAMPLES[@]}" > "slurm-logs/manifest-${PANEL}.txt"
fi
SAMPLE="${SAMPLES[$TASK_ID]}"

echo "=================================================================="
echo "job      : ${SLURM_JOB_NAME} ${SLURM_ARRAY_JOB_ID}_${TASK_ID} on $(hostname)"
echo "panel    : ${PANEL} (sample $((TASK_ID + 1))/${#SAMPLES[@]})"
echo "sample   : ${SAMPLE}"
echo "started  : $(date -Is)"
echo "=================================================================="
nvidia-smi || true

# --- convert ------------------------------------------------------------------------------------------------------
set +e
srun "$PY" "$SCRIPT" \
    --sample-dir "$SAMPLE" \
    --raw-root "$RAW_ROOT" \
    --out-root "$OUT_ROOT" \
    --num-workers "$NUM_WORKERS" \
    ${EXTRA_ARGS:-}
STATUS=$?
set -e

# A SIGKILL (OOM, exit 137), a SLURM TIMEOUT, or a bad --sample-dir kills the worker before its own try/except can
# log anything. Record those here; any other failure was already logged by the python script (exit 1).
if [[ "$STATUS" -ne 0 && "$STATUS" -ne "$EXIT_FAILURE_LOGGED" ]]; then
    REL="${SAMPLE#"$RAW_ROOT"/}"
    case "$STATUS" in
        137) REASON="SIGKILL (exit 137) - almost certainly OOM; check MaxRSS via sacct" ;;
        124|143) REASON="terminated (exit ${STATUS}) - likely SLURM TIMEOUT" ;;
        *) REASON="worker exited ${STATUS} without logging; see ${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err" ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$(cut -d/ -f1 <<< "$REL")" "$(cut -d/ -f2 <<< "$REL")" \
        "$(basename "$REL")" "$REASON" | flock "$FAIL_LOG" tee -a "$FAIL_LOG" > /dev/null
    echo "recorded uncaught failure (exit ${STATUS}) in ${FAIL_LOG}"
fi

echo "=================================================================="
echo "finished : $(date -Is)  exit=${STATUS}"
echo "=================================================================="
exit "$STATUS"
