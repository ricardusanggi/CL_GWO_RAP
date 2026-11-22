#!/bin/bash
# ============================================================
# Shell script untuk menjalankan main_auto_all_methods.py
# di background (nohup) dengan conda env metaWSL
# ============================================================

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null

conda activate metaWSL

cd "/mnt/d/Hibah BIMA/2026/BIMA_RAP/full_code_RAP"

nohup python3 main_auto_all_methods.py \
    > run_auto_all_methods.log 2>&1 &

PID=$!
echo "=================================="
echo "  PID     : $PID"
echo "  Log     : tail -f run_auto_all_methods.log"
echo "  Stop    : kill $PID"
echo "=================================="
