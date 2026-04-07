#!/bin/bash
# ============================================================
# Run ALL baseline experiments sequentially
# Usage:  bash scripts/run_all.sh [GPU_ID]
# ============================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU=${1:-0}

# Create logs directory
mkdir -p "${ROOT}/scripts/logs"

echo "============================================"
echo " Running all baselines on GPU: ${GPU}"
echo "============================================"

echo ""
echo "############ 1/3  KUCNet ############"
bash "${ROOT}/scripts/run_kucnet.sh" "${GPU}"

echo ""
echo "############ 2/3  AdaProp ############"
bash "${ROOT}/scripts/run_adaprop.sh" "${GPU}"

echo ""
echo "############ 3/3  MoKGR ##############"
bash "${ROOT}/scripts/run_mokgr.sh" "${GPU}"

echo ""
echo "============================================"
echo " ALL BASELINES COMPLETE"
echo "============================================"
