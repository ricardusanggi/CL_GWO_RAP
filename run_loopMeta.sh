#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate metaWSL

cd "/mnt/d/Hibah BIMA/2026/BIMA_RAP/full_code_RAP"
nohup python3 main_loopMeta.py > run_loopMeta.log 2>&1 &
echo "PID: $!"
echo "Log: tail -f run_loopMeta.log"
