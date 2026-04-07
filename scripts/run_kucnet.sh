#!/bin/bash
# ============================================================
# Run KUCNet experiments — reads from configs/kucnet.yaml
# Usage:  bash scripts/run_kucnet.sh [DATASET] [GPU_ID]
#   DATASET: dataset name (e.g. last-fm) or "all" (default: all)
#   GPU_ID:  cuda device id (default: 0)
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT}/configs/kucnet.yaml"
DATASET_ARG=${1:-all}
GPU=${2:-0}
WORKDIR="${ROOT}/KUCNet"
LOGDIR="${ROOT}/scripts/logs"
mkdir -p "${LOGDIR}"

# Parse a value from the YAML config: parse_yaml <file> <key_path>
# e.g. parse_yaml config.yaml "datasets.last-fm.K"
parse_yaml() {
    python3 -c "
import yaml, sys
with open('${1}') as f:
    cfg = yaml.safe_load(f)
keys = '${2}'.split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
"
}

run_dataset() {
    local DS=$1
    local K=$(parse_yaml "${CONFIG}" "datasets.${DS}.K")
    local SEED=$(parse_yaml "${CONFIG}" "training.seed")

    echo ""
    echo ">>> [KUCNet] Dataset: ${DS}  K=${K}  seed=${SEED}  gpu=${GPU}"
    echo "----------------------------------------"
    cd "${WORKDIR}"
    python train.py \
        --data_path "data/${DS}/" \
        --K "${K}" \
        --seed "${SEED}" \
        --gpu "${GPU}" \
        2>&1 | tee -a "${LOGDIR}/kucnet_${DS}.log"
    echo ">>> [KUCNet] Done: ${DS}"
}

echo "========================================"
echo " KUCNet — GPU: ${GPU}"
echo "========================================"

if [ "${DATASET_ARG}" = "all" ]; then
    DATASETS=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for ds in cfg['datasets']:
    print(ds)
")
    for DS in ${DATASETS}; do
        run_dataset "${DS}"
    done
else
    run_dataset "${DATASET_ARG}"
fi

echo ""
echo "========================================"
echo " KUCNet — Finished"
echo "========================================"
