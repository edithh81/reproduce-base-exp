#!/bin/bash
# ============================================================
# Run AdaProp experiments — reads from configs/adaprop.yaml
# Usage:  bash scripts/run_adaprop.sh [DATASET] [GPU_ID]
#   DATASET: dataset name (e.g. last-fm) or "all" (default: all)
#   GPU_ID:  cuda device id (default: 0)
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT}/configs/adaprop.yaml"
DATASET_ARG=${1:-all}
GPU=${2:-0}
WORKDIR="${ROOT}/AdaProp_RecSys"
LOGDIR="${ROOT}/scripts/logs"
mkdir -p "${LOGDIR}"

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
    local TOPK=$(parse_yaml "${CONFIG}" "datasets.${DS}.topk")
    local LAYERS=$(parse_yaml "${CONFIG}" "datasets.${DS}.n_layer")
    local SEED=$(parse_yaml "${CONFIG}" "training.seed")
    local EPOCH=$(parse_yaml "${CONFIG}" "training.epoch")
    local SCHEDULER=$(parse_yaml "${CONFIG}" "training.scheduler")
    local TAU=$(parse_yaml "${CONFIG}" "adaprop.tau")
    local PPR_TOPK=$(parse_yaml "${CONFIG}" "adaprop.ppr_topk")

    echo ""
    echo ">>> [AdaProp] Dataset: ${DS}  topk=${TOPK}  layers=${LAYERS}  tau=${TAU}"
    echo "----------------------------------------"
    cd "${WORKDIR}"
    python train.py \
        --data_path "${ROOT}/data/${DS}/" \
        --topk "${TOPK}" \
        --layers "${LAYERS}" \
        --tau "${TAU}" \
        --ppr_topk "${PPR_TOPK}" \
        --seed "${SEED}" \
        --epoch "${EPOCH}" \
        --scheduler "${SCHEDULER}" \
        --gpu "${GPU}" \
        2>&1 | tee -a "${LOGDIR}/adaprop_${DS}.log"
    echo ">>> [AdaProp] Done: ${DS}"
}

echo "========================================"
echo " AdaProp — GPU: ${GPU}"
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
echo " AdaProp — Finished"
echo "========================================"
