#!/usr/bin/env python
# ================================================================
# AUTO-RUN 2-FASE SEMUA METODE METAHEURISTIK – SCENARIO G
#
# FASE 1 – Pencarian Populasi Terbaik
#   Jalankan pop = [10,20,...,100] dengan iter tetap (FIXED_ITER)
#   Pilih pop dengan akurasi Skenario G tertinggi (tanpa memperhatikan std)
#
# FASE 2 – Pencarian Iterasi Terbaik
#   Jalankan iter = [10,20,...,100] dengan pop terbaik dari Fase 1
#   Pilih iter dengan akurasi Skenario G tertinggi
#
# 7 Metode yang dijalankan (berurutan, otomatis):
#   AMGWO_GenliangLi | BFLGWO_YongWang | CL_GWO (PROPOSED)
#   CLPSO_Liang      | GWO             | PSO    | RLCGWO_ZhengpengHu
# ================================================================

import os
import sys
import csv
import json
import time
import copy
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
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
# KONFIGURASI UTAMA
# ================================================================

# Fase 1: variasi populasi (iter dikunci di FIXED_ITER)
POPULATION_SEARCH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
FIXED_ITER        = 10      # iterasi tetap selama pencarian populasi

# Fase 2: variasi iterasi (pop dikunci dari hasil Fase 1)
ITER_SEARCH       = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# (pop terbaik ditentukan otomatis dari Fase 1)

# Skenario fitness & skenario evaluasi penentu "terbaik"
FITNESS_SCENARIO  = "G"
BEST_SCENARIO_KEY = "G_selected_only"   # kolom di final_classification_results.csv

# ================================================================
# DEFINISI METODE
# RESULT_DIR menggunakan path absolut agar aman saat dijalankan
# dari direktori manapun.
# ================================================================
METHOD_CONFIGS = [
    {
        "META_ALGO":       "CL_GWO",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_CLGWO_auto"),
        "EXPERIMENT_NAME": "cnn_CL_GWO_hybrid",
    },
    {
        "META_ALGO":       "CLPSO_Liang",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_CLPSO_Liang_auto"),
        "EXPERIMENT_NAME": "cnn_CLPSO_Liang_hybrid",
    },
    {
        "META_ALGO":       "GWO",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_GWO_auto"),
        "EXPERIMENT_NAME": "cnn_GWO_hybrid",
    },
    {
        "META_ALGO":       "PSO",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_PSO_auto"),
        "EXPERIMENT_NAME": "cnn_PSO_hybrid",
    },
    {
        "META_ALGO":       "RLCGWO_ZhengpengHu",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_RLCGWO_auto"),
        "EXPERIMENT_NAME": "cnn_RLCGWO_ZhengpengHu_hybrid",
    },
    {
        "META_ALGO":       "AMGWO_GenliangLi",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_AMGWO_auto"),
        "EXPERIMENT_NAME": "cnn_AMGWO_GenliangLi_hybrid",
    },
    {
        "META_ALGO":       "BFLGWO_YongWang",
        "RESULT_DIR":      os.path.join(BASE_DIR, "results_BFLGWO_auto"),
        "EXPERIMENT_NAME": "cnn_BFLGWO_YongWang_hybrid",
    },
]

# ================================================================
# DEVICE
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")


# ================================================================
# HELPER: BACA AKURASI DARI final_classification_results.csv
# ================================================================
def read_best_scenario_acc(exp_dir: str) -> float:
    clf_path = os.path.join(exp_dir, "final_classification_results.csv")
    if not os.path.exists(clf_path):
        return -1.0
    with open(clf_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scenario"] == BEST_SCENARIO_KEY:
                return float(row["mean_accuracy"])
    return -1.0


# ================================================================
# HELPER: SIMPAN / MUAT RESUME STATE
# ================================================================
def save_resume(m_cfg, phase, best_pop=None, best_iter=None):
    path = os.path.join(m_cfg["RESULT_DIR"], "_resume_state.json")
    ensure_dir(m_cfg["RESULT_DIR"])
    state = {
        "META_ALGO": m_cfg["META_ALGO"],
        "phase":     phase,
        "best_pop":  best_pop,
        "best_iter": best_iter,
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_resume(m_cfg):
    path = os.path.join(m_cfg["RESULT_DIR"], "_resume_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ================================================================
# HELPER: CEK APAKAH EKSPERIMEN SUDAH ADA (untuk skip)
# ================================================================
def find_existing_exp(result_dir: str, exp_prefix: str) -> str | None:
    """Kembalikan path folder eksperimen jika sudah ada, None jika belum."""
    if not os.path.isdir(result_dir):
        return None
    for d in sorted(os.listdir(result_dir)):
        if d.startswith(exp_prefix) and os.path.isdir(os.path.join(result_dir, d)):
            clf = os.path.join(result_dir, d, "final_classification_results.csv")
            if os.path.exists(clf):
                return os.path.join(result_dir, d)
    return None


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
# SATU EKSPERIMEN
# ================================================================
def run_one_experiment(cfg: dict, X_feat: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """
    Jalankan satu eksperimen.
    Return: (best_scenario_G_acc, exp_dir)
    """
    pop_size = cfg["POP_SIZE"]
    max_iter = cfg["MAX_ITER"]
    algo     = cfg["META_ALGO"]

    ensure_dir(cfg["RESULT_DIR"])
    exp_name_full = f"{cfg['EXPERIMENT_NAME']}_pop{pop_size}_iter{max_iter}"
    exp_dir = os.path.join(cfg["RESULT_DIR"], f"{exp_name_full}_{now()}")
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
    print(f"\n[RUN] {algo} | pop={pop_size} | iter={max_iter}")
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

    acc_g   = read_best_scenario_acc(exp_dir)
    opt_acc = -best.target.fitness

    print(f"      BestFit={opt_acc:.6f} | AccSkenG={acc_g:.6f} | "
          f"waktu={elapsed/60:.1f}mnt | dir={os.path.basename(exp_dir)}")

    return acc_g, exp_dir


# ================================================================
# HELPER: BUAT CFG UNTUK SATU RUN
# Deep-copy agar tidak ada kontaminasi antar eksperimen.
# EXPERIMENT_NAME tetap nama dasar; run_one_experiment yang akan
# menambahkan _pop{n}_iter{n} pada nama folder.
# ================================================================
def _make_cfg(cfg_base: dict, m_cfg: dict, pop: int, itr: int) -> dict:
    cfg = copy.deepcopy(cfg_base)
    cfg.update(m_cfg)
    cfg["POP_SIZE"]         = pop
    cfg["MAX_ITER"]         = itr
    cfg["FITNESS_SCENARIO"] = FITNESS_SCENARIO
    cfg["EXPERIMENT_NAME"]  = m_cfg["EXPERIMENT_NAME"]
    return cfg


# ================================================================
# FASE 1 – PENCARIAN POPULASI TERBAIK
# ================================================================
def phase1_search_population(m_cfg: dict, cfg_base: dict,
                             X_feat: np.ndarray, y: np.ndarray,
                             global_log: list) -> int:
    algo = m_cfg["META_ALGO"]
    sep  = "─" * 70

    global_log.append(f"\n{sep}")
    global_log.append(f"  FASE 1 – {algo} | Pencarian Populasi Terbaik (iter={FIXED_ITER})")
    global_log.append(sep)
    print(f"\n{'='*70}")
    print(f"  FASE 1 – {algo} | Pencarian Populasi (iter={FIXED_ITER})")
    print(f"{'='*70}")

    results: dict[int, float] = {}

    for pop in POPULATION_SEARCH:
        exp_prefix = f"{m_cfg['EXPERIMENT_NAME']}_pop{pop}_iter{FIXED_ITER}"

        existing = find_existing_exp(m_cfg["RESULT_DIR"], exp_prefix)
        if existing:
            acc_g = read_best_scenario_acc(existing)
            results[pop] = acc_g
            msg = f"  [SKIP] {algo} pop={pop:3d} iter={FIXED_ITER} → AccG={acc_g:.6f} (sudah ada)"
            global_log.append(msg); print(msg)
            continue

        cfg = _make_cfg(cfg_base, m_cfg, pop=pop, itr=FIXED_ITER)
        try:
            acc_g, _ = run_one_experiment(cfg, X_feat, y)
            results[pop] = acc_g
            msg = f"  [OK]   {algo} pop={pop:3d} iter={FIXED_ITER} → AccG={acc_g:.6f}"
        except Exception as e:
            results[pop] = -1.0
            msg = f"  [FAIL] {algo} pop={pop:3d} iter={FIXED_ITER} → {e}"
            traceback.print_exc()

        global_log.append(msg); print(msg)

    if not results:
        raise RuntimeError(f"[{algo}] Semua run Fase 1 gagal, tidak ada hasil.")

    best_pop = max(results, key=lambda p: results[p])
    best_acc = results[best_pop]

    summary = (f"\n  ★ BEST POP  {algo}: pop={best_pop} | AccScenG={best_acc:.6f}\n"
               f"  Ranking: {sorted(results.items(), key=lambda x: -x[1])}")
    global_log.append(summary); print(summary)

    _save_phase_table(
        m_cfg["RESULT_DIR"], "phase1_population_search.csv",
        results, "pop_size", "acc_scenario_G", best_key=best_pop
    )

    return best_pop


# ================================================================
# FASE 2 – PENCARIAN ITERASI TERBAIK
# ================================================================
def phase2_search_iter(m_cfg: dict, cfg_base: dict,
                       X_feat: np.ndarray, y: np.ndarray,
                       best_pop: int, global_log: list) -> int:
    algo = m_cfg["META_ALGO"]
    sep  = "─" * 70

    global_log.append(f"\n{sep}")
    global_log.append(f"  FASE 2 – {algo} | Pencarian Iterasi Terbaik (pop={best_pop})")
    global_log.append(sep)
    print(f"\n{'='*70}")
    print(f"  FASE 2 – {algo} | Pencarian Iterasi (pop={best_pop})")
    print(f"{'='*70}")

    results: dict[int, float] = {}

    for itr in ITER_SEARCH:
        exp_prefix = f"{m_cfg['EXPERIMENT_NAME']}_pop{best_pop}_iter{itr}"

        existing = find_existing_exp(m_cfg["RESULT_DIR"], exp_prefix)
        if existing:
            acc_g = read_best_scenario_acc(existing)
            results[itr] = acc_g
            msg = f"  [SKIP] {algo} pop={best_pop} iter={itr:3d} → AccG={acc_g:.6f} (sudah ada)"
            global_log.append(msg); print(msg)
            continue

        cfg = _make_cfg(cfg_base, m_cfg, pop=best_pop, itr=itr)
        try:
            acc_g, _ = run_one_experiment(cfg, X_feat, y)
            results[itr] = acc_g
            msg = f"  [OK]   {algo} pop={best_pop} iter={itr:3d} → AccG={acc_g:.6f}"
        except Exception as e:
            results[itr] = -1.0
            msg = f"  [FAIL] {algo} pop={best_pop} iter={itr:3d} → {e}"
            traceback.print_exc()

        global_log.append(msg); print(msg)

    if not results:
        raise RuntimeError(f"[{algo}] Semua run Fase 2 gagal, tidak ada hasil.")

    best_iter = max(results, key=lambda i: results[i])
    best_acc  = results[best_iter]

    summary = (f"\n  ★ BEST ITER {algo}: iter={best_iter} (pop={best_pop}) | AccScenG={best_acc:.6f}\n"
               f"  Ranking: {sorted(results.items(), key=lambda x: -x[1])}")
    global_log.append(summary); print(summary)

    _save_phase_table(
        m_cfg["RESULT_DIR"], "phase2_iter_search.csv",
        results, "max_iter", "acc_scenario_G", best_key=best_iter
    )

    return best_iter


# ================================================================
# HELPER: SIMPAN TABEL HASIL FASE
# ================================================================
def _save_phase_table(result_dir, filename, results_dict,
                      key_col, val_col, best_key):
    ensure_dir(result_dir)
    path = os.path.join(result_dir, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key_col, val_col, "is_best"])
        for k in sorted(results_dict):
            w.writerow([k, results_dict[k], "YES" if k == best_key else ""])
    print(f"  [SAVED] {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    global_log  = []
    all_summary = {}
    t_global    = time.time()

    cfg_base = build_config()
    cfg_base["FITNESS_SCENARIO"] = FITNESS_SCENARIO

    print("[INIT] Membangun CNN backbone ...")
    extractor = CNNFeatureExtractor(
        img_size=cfg_base["IMG_SIZE"],
        backbone=cfg_base["BACKBONE"],
        device=DEVICE
    )
    backbone, preprocess, feat_dim = extractor.build()

    print("[INIT] Memuat dataset ...")
    X_img, y = load_images(cfg_base["DATASET_DIR"], preprocess)

    print("[INIT] Ekstraksi fitur CNN ...")
    X_feat = backbone.predict(X_img, batch_size=512)
    del X_img   # bebaskan RAM; gambar tidak dipakai lagi setelah ini
    print(f"[INIT] X_feat: {X_feat.shape} | Kelas: {len(set(y))}\n")

    header = (
        f"{'='*70}\n"
        f"  AUTO-RUN 2-FASE | {len(METHOD_CONFIGS)} METODE\n"
        f"  Fase 1 : pop={POPULATION_SEARCH}  iter={FIXED_ITER} (tetap)\n"
        f"  Fase 2 : pop=best(Fase1)  iter={ITER_SEARCH}\n"
        f"  Fitness: Scenario {FITNESS_SCENARIO}  |  Evaluasi best: {BEST_SCENARIO_KEY}\n"
        f"{'='*70}"
    )
    global_log.append(header)
    print(header)

    for m_cfg in METHOD_CONFIGS:
        algo     = m_cfg["META_ALGO"]
        t_method = time.time()

        try:
            resume = load_resume(m_cfg)
            if resume and resume.get("phase") == "done":
                best_pop  = resume["best_pop"]
                best_iter = resume["best_iter"]
                msg = f"\n  [RESUME] {algo}: best_pop={best_pop} best_iter={best_iter} (sudah selesai)"
                global_log.append(msg); print(msg)
            else:
                if resume and resume.get("best_pop") and resume.get("phase") == "fase2":
                    best_pop = resume["best_pop"]
                    msg = f"\n  [RESUME] {algo}: Fase 1 sudah selesai → best_pop={best_pop}"
                    global_log.append(msg); print(msg)
                else:
                    best_pop = phase1_search_population(m_cfg, cfg_base, X_feat, y, global_log)
                    save_resume(m_cfg, "fase2", best_pop=best_pop)

                best_iter = phase2_search_iter(m_cfg, cfg_base, X_feat, y, best_pop, global_log)
                save_resume(m_cfg, "done", best_pop=best_pop, best_iter=best_iter)

            elapsed_m       = (time.time() - t_method) / 60
            best_exp_prefix = f"{m_cfg['EXPERIMENT_NAME']}_pop{best_pop}_iter{best_iter}"
            best_exp_dir    = find_existing_exp(m_cfg["RESULT_DIR"], best_exp_prefix)
            best_acc_final  = read_best_scenario_acc(best_exp_dir) if best_exp_dir else -1.0

            all_summary[algo] = {
                "best_pop":        best_pop,
                "best_iter":       best_iter,
                "best_acc_scen_G": best_acc_final,
                "elapsed_mnt":     round(elapsed_m, 1),
            }

            row = (f"  ✓ {algo:22s} | best_pop={best_pop:3d} | best_iter={best_iter:3d} "
                   f"| AccG={best_acc_final:.6f} | {elapsed_m:.1f} mnt")
            global_log.append(row); print(row)

        except Exception as e:
            msg = f"  ✗ GAGAL {algo}: {e}"
            global_log.append(msg); print(msg)
            traceback.print_exc()

    total_h = (time.time() - t_global) / 3600
    footer_lines = [
        f"\n{'='*70}",
        f"  SELESAI | Total waktu: {total_h:.2f} jam",
        f"  {'METODE':<24} {'BEST POP':>9} {'BEST ITER':>10} {'ACC Scen G':>12}",
        f"  {'─'*60}",
    ]
    for algo, s in all_summary.items():
        mark = "★" if algo == "CL_GWO" else " "
        footer_lines.append(
            f"{mark} {algo:<23} {s['best_pop']:>9} {s['best_iter']:>10} {s['best_acc_scen_G']:>12.6f}"
        )
    footer_lines.append(f"{'='*70}")
    for ln in footer_lines:
        global_log.append(ln); print(ln)

    summary_json = os.path.join(BASE_DIR, "best_params_summary.json")
    with open(summary_json, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\n  Summary params → {summary_json}")

    log_path = os.path.join(BASE_DIR, "run_auto_all_methods.log")
    with open(log_path, "w") as f:
        f.write("\n".join(global_log))
    print(f"  Log lengkap    → {log_path}")


if __name__ == "__main__":
    main()
