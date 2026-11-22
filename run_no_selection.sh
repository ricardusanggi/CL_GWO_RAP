#!/bin/bash
# run_no_selection.sh
# Jalankan Transfer Learning + MLP tanpa seleksi fitur
# Double-click di file manager atau jalankan: bash run_no_selection.sh

# Aktifkan conda (coba anaconda3 dulu, fallback ke miniconda3)
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null

conda activate metaWSL

cd "/mnt/d/Hibah BIMA/2026/BIMA_RAP/full_code_RAP"

LOG_FILE="run_no_selection.log"

echo "=================================================="
echo "  Transfer Learning + MLP (No Feature Selection)"
echo "  Start : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log   : $LOG_FILE"
echo "=================================================="

nohup python3 main_no_selection.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "PID   : $PID"
echo "Monitor log dengan:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Proses berjalan di background."
echo "Tutup terminal aman — proses tetap berjalan."
