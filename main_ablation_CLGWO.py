#!/usr/bin/env python
# ================================================================
# ABLATION STUDY – CL-GWO
# Memisahkan kontribusi dimension-wise learning (DWL) dari adaptive
# leader selection (ALS), untuk menjawab Reviewer 2 poin 5:
# "Belum ada ablasi yang memisahkan kontribusi dimension-wise
#  learning dari adaptive leader selection secara terpisah,
#  sehingga klaim bahwa kombinasi keduanya penting belum
#  dibuktikan secara empiris."
#
# Varian yang dibandingkan (pop=90, iter=60 TETAP untuk semua varian,
# sama persis dengan konfigurasi CL-GWO terbaik di Table 4/Bagian 4.3;
# hanya META_ALGO yang berbeda, agar perbandingan adil):
#   GWO             – baseline standar (skema kontrol-a berbeda, lihat catatan
#                     di ABLATION_VARIANTS; bukan bagian dari isolasi DWL/ALS)
#   CL_GWO_NoALS    – dimension-wise learning (DWL) ON, adaptive leader (ALS) OFF
#   CL_GWO_NoDWL    – adaptive leader (ALS) ON, dimension-wise learning OFF
#   CL_GWO          – full (proposed)
#
# CL_GWO, CL_GWO_NoDWL, CL_GWO_NoALS berbagi w_min=0.2, w_max=2.0, max_flag=7
# secara identik (lihat core/optimizer_factory.py) sehingga satu-satunya
# variabel yang berbeda antar ketiganya adalah mekanisme yang dihilangkan.
#
# Skala dipangkas (2 skenario x 5 repeat, bukan 4 skenario x 10 repeat)
# untuk menekan biaya komputasi dari ~4.2 hari menjadi ~1 hari, sambil
# tetap punya cukup sampel (n=5/grup) untuk ANOVA/Wilcoxon kasar.
# Skenario dipilih agar representatif: E (semua fitur asli + fitur
# sintesis) dan G (hanya fitur terpilih, seleksi murni).
#
# Semua output disimpan di folder baru RESULT_DIR (nama menjelaskan
# tujuannya) agar jelas untuk apa folder ini dibuat:
#   ablation_run.log        – log lengkap proses (realtime, append-only)
#   MANIFEST.txt            – tujuan, konfigurasi, dan waktu pembuatan run
#   ablation_summary.csv    – satu baris per run (ditulis incremental)
#   ablation_stats.csv      – mean/std + ANOVA per skenario (di akhir)
#   ablation_<varian>_scen<X>_seed<N>_.../  – folder hasil mealpy per run
# ================================================================

import os
import sys
import csv
import time
import traceback
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from mealpy.utils.space import FloatVar

from config import build_config
from utils.io import ensure_dir, now
from utils.logger import UnifiedLogger

from core.cnn_feature_extractor import CNNFeatureExtractor
from core.feature_constructor import FeatureConstructor
from core.fitness import Fitness
from core.objective import MetaObjective
from core.optimizer_factory import create_optimizer
from core.final_classification import run_classification

# ================================================================
# KONFIGURASI ABLASI
# ================================================================

# Skenario fitness yang diuji: subset representatif (bukan F,G,A,E penuh)
# untuk menekan biaya komputasi ke ~1 hari.
ABLATION_SCENARIOS = ["E", "G"]

SCENARIO_MAP = {
    "A": "A_selected_plus_f_selected",
    "B": "B_discarded_plus_f_discarded",
    "C": "C_all_plus_f_selected",
    "D": "D_all_plus_f_discarded",
    "E": "E_all_plus_f_selected_plus_f_discarded",
    "F": "F_all_features_only",
    "G": "G_selected_only",
    "H": "H_discarded_only",
}

# Varian yang dibandingkan (nama tampilan -> META_ALGO).
# GWO baseline dipertahankan untuk konteks, TAPI perlu dicatat: OriginalGWO
# memakai skema kontrol-a standar (linear 2->0), bukan w_min/w_max/max_flag
# ala CL_GWO, sehingga baseline ini TIDAK sepenuhnya "satu variabel berbeda"
# terhadap ketiga varian CL_GWO lainnya -- ia tetap dilaporkan sebagai acuan
# standar (sesuai Bagian 4), bukan sebagai bagian dari isolasi DWL/ALS.
# Isolasi murni DWL vs ALS ada pada trio CL_GWO / CL_GWO_NoDWL / CL_GWO_NoALS,
# yang berbagi w_min=0.2, w_max=2.0, max_flag=7 secara identik (lihat
# core/optimizer_factory.py::create_optimizer).
ABLATION_VARIANTS = {
    "GWO":            "GWO",
    "CL_GWO_NoALS":   "CL_GWO_NoALS",   # DWL ON, Adaptive Leader Selection OFF
    "CL_GWO_NoDWL":   "CL_GWO_NoDWL",   # ALS ON, Dimension-Wise Learning OFF
    "CL_GWO":         "CL_GWO",         # full (proposed)
}

# pop=90, iter=60: parameter terbaik CL-GWO hasil tuning Phase 1/Phase 2
# (Bagian 3.6), dilaporkan di Table 4 (Bagian 4.3). Dikunci SAMA untuk semua
# varian ablasi agar satu-satunya variabel yang berbeda benar-benar hanya
# mekanisme yang dihilangkan (DWL atau ALS).
ABLATION_POP_SIZE = 90
ABLATION_MAX_ITER = 60

# Jumlah run independen per (varian, skenario), seed berbeda tiap run
REPEATS = 5
BASE_SEED = 42

# Folder hasil bernama jelas agar tahu untuk apa dibuat tanpa perlu membaca kode.
RESULT_DIR = "results_ablation_Reviewer2_DWL_vs_ALS"
LOG_PATH = None  # diisi di main() setelah RESULT_DIR dipastikan ada


# ================================================================
# LOGGING: cetak ke stdout SEKALIGUS tulis realtime ke file log,
# supaya progres tidak hilang meski proses berjalan berjam-jam
# dan terhenti di tengah jalan (crash / dihentikan manual).
# ================================================================
def log(msg: str = ""):
    print(msg)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def write_manifest():
    path = os.path.join(RESULT_DIR, "MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "ABLATION STUDY - CL-GWO: dimension-wise learning (DWL) vs "
            "adaptive leader selection (ALS)\n"
            "================================================================\n"
            f"Dibuat        : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            "Tujuan        : Menjawab Reviewer 2 poin 5 - memisahkan kontribusi\n"
            "                dimension-wise learning dari adaptive leader selection\n"
            "                secara terpisah, untuk membuktikan secara empiris bahwa\n"
            "                kombinasi keduanya (CL-GWO penuh) memang penting.\n"
            f"Skenario      : {ABLATION_SCENARIOS}\n"
            f"Varian        : {list(ABLATION_VARIANTS.keys())}\n"
            f"pop / iter    : {ABLATION_POP_SIZE} / {ABLATION_MAX_ITER} "
            "(identik dengan CL-GWO terbaik, Table 4/Bagian 4.3)\n"
            f"w_min/w_max   : 0.2 / 2.0 (default mealpy CL_GWO, dipakai eksperimen asli)\n"
            f"max_flag      : 7 (bookkeeping saja, tidak mengubah pencarian - lihat\n"
            "                catatan stagnation-flag di optimizer_factory.py)\n"
            f"repeats       : {REPEATS} run/varian/skenario, seed = {BASE_SEED}..{BASE_SEED + REPEATS - 1}\n"
            f"total run     : {len(ABLATION_VARIANTS) * len(ABLATION_SCENARIOS) * REPEATS}\n"
            "\n"
            "Isi folder ini:\n"
            "  ablation_run.log      - log lengkap proses, realtime\n"
            "  ablation_summary.csv  - satu baris per run (accuracy, waktu, status)\n"
            "  ablation_stats.csv    - mean/std + ANOVA per skenario (ditulis di akhir)\n"
            "  ablation_<varian>_scen<X>_seed<N>_.../ - folder hasil mealpy per run\n"
        )
    log(f"[SAVED] Manifest → {path}")


# ================================================================
# DEVICE
# ================================================================
def init_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[INFO] Device : {device}")
    if device == "cuda":
        log(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
    return device


# ================================================================
# HELPER: BACA AKURASI DARI final_classification_results.csv
# ================================================================
def read_best_scenario_acc(exp_dir: str, best_scenario_key: str) -> float:
    clf_path = os.path.join(exp_dir, "final_classification_results.csv")
    if not os.path.exists(clf_path):
        return -1.0
    with open(clf_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scenario"] == best_scenario_key:
                return float(row["mean_accuracy"])
    return -1.0


# ================================================================
# LOAD IMAGES
# ================================================================
def load_images(dataset_dir, preprocess):
    X, y = [], []
    for label in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            try:
                img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
                img = preprocess(img)
                X.append(img.numpy().transpose(1, 2, 0))
                y.append(label)
            except Exception:
                continue
    return np.asarray(X), LabelEncoder().fit_transform(y)


# ================================================================
# SATU RUN ABLASI
# ================================================================
def run_one_ablation(cfg: dict, X_feat: np.ndarray, y: np.ndarray,
                      best_scenario_key: str) -> tuple[float, float]:
    pop_size = cfg["POP_SIZE"]
    max_iter = cfg["MAX_ITER"]
    algo     = cfg["META_ALGO"]
    sc       = cfg["FITNESS_SCENARIO"]
    seed     = cfg["RANDOM_STATE"]

    ensure_dir(cfg["RESULT_DIR"])
    exp_dir = os.path.join(cfg["RESULT_DIR"], f"{cfg['EXPERIMENT_NAME']}_{now()}")
    ensure_dir(exp_dir)

    feat_dim    = X_feat.shape[1]
    constructor = FeatureConstructor(feat_dim)
    fitness     = Fitness(X_feat, y, cfg)

    problem = {
        "obj_func": MetaObjective(X_feat, fitness, constructor, cfg, alpha=0.3),
        "bounds":   FloatVar(lb=[0.0] * (2 * feat_dim), ub=[1.0] * (2 * feat_dim)),
        "minmax":   "min",
        "save_population": True,
    }

    optimizer = create_optimizer(cfg)

    t0 = time.time()
    log(f"\n[RUN] {algo} | scen={sc} | seed={seed} | pop={pop_size} | iter={max_iter}")
    best = optimizer.solve(problem)
    elapsed = time.time() - t0

    logger = UnifiedLogger(exp_dir, constructor, cfg)
    logger.log_from_history(optimizer)
    logger.save_best_only()
    logger.save_population_history(optimizer)

    decoded = constructor.decode(best.solution)
    if decoded is None:
        raise RuntimeError(f"[{algo}] Best solution tidak valid")

    f_sel, f_dis = constructor.construct_features_only(X_feat, decoded)
    final_results = run_classification(
        X_feat=X_feat, y=y, decoded=decoded,
        f_sel=f_sel, f_dis=f_dis, cfg=cfg,
        mode=cfg["CLASSIFIER_MODE"].lower()
    )
    logger.save_final_classification_results(final_results)

    acc = read_best_scenario_acc(exp_dir, best_scenario_key)

    log(f"      Acc[{best_scenario_key}]={acc:.6f} | waktu={elapsed/60:.1f}mnt | "
        f"dir={os.path.basename(exp_dir)}")

    return acc, elapsed


# ================================================================
# HELPER: TULIS/APPEND SATU BARIS KE ablation_summary.csv SEGERA
# (bukan hanya di akhir), supaya progres tidak hilang jika proses
# berhenti di tengah jalan.
# ================================================================
SUMMARY_FIELDS = ["fitness_scenario", "variant", "seed", "accuracy",
                   "elapsed_sec", "status"]


def append_summary_row(path: str, row: dict, write_header: bool):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ================================================================
# MAIN
# ================================================================
def main():
    global LOG_PATH

    ensure_dir(RESULT_DIR)
    LOG_PATH = os.path.join(RESULT_DIR, "ablation_run.log")

    write_manifest()

    t_global = time.time()
    device = init_device()

    cfg_base = build_config()

    log("[INIT] Membangun CNN backbone ...")
    extractor = CNNFeatureExtractor(
        img_size=cfg_base["IMG_SIZE"],
        backbone=cfg_base["BACKBONE"],
        device=device
    )
    backbone, preprocess, feat_dim = extractor.build()

    log("[INIT] Memuat dataset ...")
    X_img, y = load_images(cfg_base["DATASET_DIR"], preprocess)

    log("[INIT] Ekstraksi fitur CNN ...")
    X_feat = backbone.predict(X_img, batch_size=512)
    log(f"[INIT] X_feat: {X_feat.shape} | Kelas: {len(set(y))}\n")

    total_runs = len(ABLATION_VARIANTS) * len(ABLATION_SCENARIOS) * REPEATS
    header = (
        f"{'='*70}\n"
        f"  ABLATION STUDY - CL-GWO (dimension-wise learning vs. adaptive leader)\n"
        f"  Reviewer 2, poin 5\n"
        f"  Skenario : {ABLATION_SCENARIOS}\n"
        f"  Varian   : {list(ABLATION_VARIANTS.keys())}\n"
        f"  pop={ABLATION_POP_SIZE}  iter={ABLATION_MAX_ITER}  repeats={REPEATS}\n"
        f"  Total run: {total_runs}\n"
        f"{'='*70}"
    )
    log(header)

    summary_rows = []
    per_run_path = os.path.join(RESULT_DIR, "ablation_summary.csv")
    header_written = os.path.exists(per_run_path)

    run_idx = 0
    for sc in ABLATION_SCENARIOS:
        bsk = SCENARIO_MAP[sc]

        for variant_name, algo in ABLATION_VARIANTS.items():
            for r in range(REPEATS):
                run_idx += 1
                seed = BASE_SEED + r

                cfg = cfg_base.copy()
                cfg["FITNESS_SCENARIO"] = sc
                cfg["META_ALGO"]        = algo
                cfg["POP_SIZE"]         = ABLATION_POP_SIZE
                cfg["MAX_ITER"]         = ABLATION_MAX_ITER
                cfg["RANDOM_STATE"]     = seed
                cfg["RESULT_DIR"]       = RESULT_DIR
                cfg["EXPERIMENT_NAME"]  = f"ablation_{variant_name}_scen{sc}_seed{seed}"

                log(f"\n[{run_idx}/{total_runs}] scen={sc} variant={variant_name} seed={seed}")

                try:
                    acc, elapsed = run_one_ablation(cfg, X_feat, y, bsk)
                    status = "OK"
                except Exception as e:
                    acc, elapsed = -1.0, -1.0
                    status = f"FAIL: {e}"
                    log(f"      [FAIL] {e}")
                    log(traceback.format_exc())

                row = {
                    "fitness_scenario": sc,
                    "variant": variant_name,
                    "seed": seed,
                    "accuracy": acc,
                    "elapsed_sec": round(elapsed, 1) if elapsed >= 0 else -1,
                    "status": status,
                }
                summary_rows.append(row)

                # Tulis segera (bukan hanya di akhir) agar progres tidak hilang.
                append_summary_row(per_run_path, row, write_header=not header_written)
                header_written = True

    log(f"\n[SAVED] Per-run results → {per_run_path}")

    # ── Statistik mean/std + ANOVA per skenario ──
    stats_path = os.path.join(RESULT_DIR, "ablation_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fitness_scenario", "variant", "n", "mean_accuracy",
                    "std_accuracy", "anova_F", "anova_p_vs_other_variants"])

        for sc in ABLATION_SCENARIOS:
            per_variant_acc = {}
            for variant_name in ABLATION_VARIANTS:
                accs = [row["accuracy"] for row in summary_rows
                        if row["fitness_scenario"] == sc
                        and row["variant"] == variant_name
                        and row["accuracy"] >= 0]
                per_variant_acc[variant_name] = accs

            groups = [v for v in per_variant_acc.values() if len(v) >= 2]
            if len(groups) >= 2:
                f_stat, p_val = f_oneway(*groups)
            else:
                f_stat, p_val = float("nan"), float("nan")

            for variant_name, accs in per_variant_acc.items():
                mean_acc = float(np.mean(accs)) if accs else float("nan")
                std_acc  = float(np.std(accs)) if accs else float("nan")
                w.writerow([sc, variant_name, len(accs), round(mean_acc, 6),
                            round(std_acc, 6), round(f_stat, 4), round(p_val, 6)])

    log(f"[SAVED] Ablation stats (mean/std + ANOVA) → {stats_path}")

    total_h = (time.time() - t_global) / 3600
    log(f"\n[DONE] Ablation study selesai | Total waktu: {total_h:.2f} jam")


if __name__ == "__main__":
    main()
