#!/bin/bash
# ============================================================
# Run MoKGR experiments — reads from configs/mokgr.yaml
# Usage:  bash scripts/run_mokgr.sh [DATASET] [GPU_ID]
#   DATASET: dataset name (e.g. last-fm) or "all" (default: all)
#   GPU_ID:  cuda device id (default: 0)
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT}/configs/mokgr.yaml"
DATASET_ARG=${1:-all}
GPU=${2:-0}
WORKDIR="${ROOT}/MoKGR-recsys/MoKGR_RecSys"
LOGDIR="${ROOT}/scripts/logs"
mkdir -p "${LOGDIR}"

parse_yaml() {
    python3 -c "
import yaml, sys
with open('${1}') as f:
    cfg = yaml.safe_load(f)
keys = '${2}'.split('.')
val = cfg
try:
    for k in keys:
        val = val[k]
    print(val)
except (KeyError, TypeError):
    sys.exit(1)
" 2>/dev/null
}

run_dataset() {
    local DS=$1

    # Per-dataset params
    local MIN_HOP=$(parse_yaml "${CONFIG}" "datasets.${DS}.min_hop")
    local MAX_HOP=$(parse_yaml "${CONFIG}" "datasets.${DS}.max_hop")

    # Training
    local SEED=$(parse_yaml "${CONFIG}" "training.seed")
    local EPOCH=$(parse_yaml "${CONFIG}" "training.epoch")
    local TAU=$(parse_yaml "${CONFIG}" "training.tau")

    # Gate
    local GATE_THRESH=$(parse_yaml "${CONFIG}" "gate.gate_threshold")
    local ACTIVE_GATE=$(parse_yaml "${CONFIG}" "gate.active_gate")

    # PPR
    local ACTIVE_PPR=$(parse_yaml "${CONFIG}" "ppr.active_PPR")
    local SAMP_PCT=$(parse_yaml "${CONFIG}" "ppr.sampling_percentage")
    local PPR_ALPHA=$(parse_yaml "${CONFIG}" "ppr.PPR_alpha")
    local MAX_ITER=$(parse_yaml "${CONFIG}" "ppr.max_iter")
    local PPR_BATCH=$(parse_yaml "${CONFIG}" "ppr.ppr_batch_size")

    # MoE hop
    local NUM_EXPERTS=$(parse_yaml "${CONFIG}" "moe_hop.num_experts")
    local LAMBDA_IMP=$(parse_yaml "${CONFIG}" "moe_hop.lambda_importance")
    local LAMBDA_LOAD=$(parse_yaml "${CONFIG}" "moe_hop.lambda_load")
    local LAMBDA_NOISE=$(parse_yaml "${CONFIG}" "moe_hop.lambda_noise")
    local HOP_TEMP=$(parse_yaml "${CONFIG}" "moe_hop.hop_temperature")

    # MoE pruning — K values and num_pruning_experts are per-dataset; others are global
    local _N_PRUNE_DS=$(parse_yaml "${CONFIG}" "datasets.${DS}.num_pruning_experts")
    local N_PRUNE_EXP=${_N_PRUNE_DS:-$(parse_yaml "${CONFIG}" "moe_pruning.num_pruning_experts")}
    local PRUNE_TEMP=$(parse_yaml "${CONFIG}" "moe_pruning.pruning_temperature")
    local K_SRC=$(parse_yaml "${CONFIG}" "datasets.${DS}.K_source")
    local K_MIN=$(parse_yaml "${CONFIG}" "datasets.${DS}.K_min")
    local K_MAX=$(parse_yaml "${CONFIG}" "datasets.${DS}.K_max")
    local L_INFL=$(parse_yaml "${CONFIG}" "moe_pruning.l_inflection")
    local A_VAL=$(parse_yaml "${CONFIG}" "moe_pruning.a")
    local LAMBDA_IMP_P=$(parse_yaml "${CONFIG}" "moe_pruning.lambda_importance_pruning")
    local LAMBDA_NOISE_P=$(parse_yaml "${CONFIG}" "moe_pruning.lambda_noise_pruning")

    # Build optional flags
    local EXTRA_FLAGS=""
    if [ "${ACTIVE_PPR}" = "True" ] || [ "${ACTIVE_PPR}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --active_PPR"
    fi
    if [ "${ACTIVE_GATE}" = "True" ] || [ "${ACTIVE_GATE}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --active_gate"
    fi

    echo ""
    echo ">>> [MoKGR] Dataset: ${DS}  hops=${MIN_HOP}-${MAX_HOP}  experts=${NUM_EXPERTS}"
    echo "----------------------------------------"
    cd "${WORKDIR}"
    python train.py \
        --data_path "${ROOT}/data/${DS}/" \
        --seed "${SEED}" \
        --gpu "${GPU}" \
        --epoch "${EPOCH}" \
        --tau "${TAU}" \
        --gate_threshold "${GATE_THRESH}" \
        --sampling_percentage "${SAMP_PCT}" \
        --PPR_alpha "${PPR_ALPHA}" \
        --max_iter "${MAX_ITER}" \
        --ppr_batch_size "${PPR_BATCH}" \
        --num_experts "${NUM_EXPERTS}" \
        --min_hop "${MIN_HOP}" \
        --max_hop "${MAX_HOP}" \
        --lambda_importance "${LAMBDA_IMP}" \
        --lambda_load "${LAMBDA_LOAD}" \
        --lambda_noise "${LAMBDA_NOISE}" \
        --hop_temperature "${HOP_TEMP}" \
        --pruning_temperature "${PRUNE_TEMP}" \
        --K_source "${K_SRC}" \
        --K_min "${K_MIN}" \
        --K_max "${K_MAX}" \
        --l_inflection "${L_INFL}" \
        --a "${A_VAL}" \
        --num_pruning_experts "${N_PRUNE_EXP}" \
        --lambda_importance_pruning "${LAMBDA_IMP_P}" \
        --lambda_noise_pruning "${LAMBDA_NOISE_P}" \
        ${EXTRA_FLAGS} \
        2>&1 | tee -a "${LOGDIR}/mokgr_${DS}.log"
    echo ">>> [MoKGR] Done: ${DS}"
}

echo "========================================"
echo " MoKGR — GPU: ${GPU}"
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
echo " MoKGR — Finished"
echo "========================================"
