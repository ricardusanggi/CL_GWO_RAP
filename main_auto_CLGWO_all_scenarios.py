#!/usr/bin/env python
# ================================================================
# AUTO-RUN 2-FASE  –  CL_GWO (PROPOSED)
# Skenario Fitness Bergantian: A → B → C → D → E → F → G → H
#
# Untuk setiap skenario fitness (A–H):
#   FASE 1 – Cari pop terbaik (iter = FIXED_ITER tetap)
#             pop ∈ {10, 20, …, 100}
#             Evaluasi best = skenario klasifikasi yang berpadanan
#
#   FASE 2 – Cari iter terbaik (pop = hasil Fase 1)
#             iter ∈ {10, 20, …, 100}
#
# Hasil tiap skenario disimpan di:
#   results_CLGWO_scenA/  …  results_CLGWO_scenH/
# ================================================================

import os
import sys
import csv
import json
import time
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
FIXED_ITER        = 10

# Fase 2: variasi iterasi (pop dikunci dari hasil Fase 1)
ITER_SEARCH       = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# ================================================================
# PEMETAAN SKENARIO FITNESS → KOLOM EVALUASI TERBAIK
# Kolom 'scenario' di final_classification_results.csv
# ================================================================
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

# Urutan skenario yang dijalankan bergantian
FITNESS_SCENARIOS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# ================================================================
# DEFINISI PER-SKENARIO
# ================================================================
def build_scenario_configs():
    configs = []
    for sc in FITNESS_SCENARIOS:
        configs.append({
            "FITNESS_SCENARIO":  sc,
            "BEST_SCENARIO_KEY": SCENARIO_MAP[sc],
            "META_ALGO":         "CL_GWO",
            "RESULT_DIR":        f"results_CLGWO_scen{sc}",
            "EXPERIMENT_NAME":   f"cnn_CL_GWO_scen{sc}_hybrid",
        })
    return configs

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
# HELPER: SIMPAN / MUAT RESUME STATE
# ================================================================
def save_resume(result_dir: str, phase: str, fitness_sc: str,
                best_pop=None, best_iter=None):
    ensure_dir(result_dir)
    path = os.path.join(result_dir, "_resume_state.json")
    state = {
        "fitness_scenario": fitness_sc,
        "phase":            phase,
        "best_pop":         best_pop,
        "best_iter":        best_iter,
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_resume(result_dir: str):
    path = os.path.join(result_dir, "_resume_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ================================================================
# HELPER: CEK APAKAH EKSPERIMEN SUDAH ADA
# ================================================================
def find_existing_exp(result_dir: str, exp_prefix: str):
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
def run_one_experiment(cfg: dict, X_feat: np.ndarray, y: np.ndarray,
                       best_scenario_key: str) -> tuple[float, str]:
    """
    Jalankan satu eksperimen.
    Return: (acc_skenario_berpadanan, exp_dir)
    """
    pop_size = cfg["POP_SIZE"]
    max_iter = cfg["MAX_ITER"]
    algo     = cfg["META_ALGO"]
    sc       = cfg["FITNESS_SCENARIO"]

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
    print(f"\n[RUN] {algo} | scen={sc} | pop={pop_size} | iter={max_iter}")
    best = optimizer.solve(problem)
    elapsed = time.time() - t0

    # Logging
    logger = UnifiedLogger(exp_dir, constructor, cfg)
    logger.log_from_history(optimizer)
    logger.save_best_only()
    logger.save_population_history(optimizer)

    # Final classification semua skenario A–H
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
    opt_acc = -best.target.fitness

    print(f"      BestFit={opt_acc:.6f} | Acc[{best_scenario_key}]={acc:.6f} | "
          f"waktu={elapsed/60:.1f}mnt | dir={os.path.basename(exp_dir)}")

    return acc, exp_dir


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
# FASE 1 – PENCARIAN POPULASI TERBAIK
# ================================================================
def phase1_search_population(sc_cfg: dict, cfg_base: dict,
                              X_feat: np.ndarray, y: np.ndarray,
                              global_log: list) -> int:
    algo   = sc_cfg["META_ALGO"]
    sc     = sc_cfg["FITNESS_SCENARIO"]
    bsk    = sc_cfg["BEST_SCENARIO_KEY"]
    sep    = "─" * 70

    header = (f"\n{sep}\n"
              f"  FASE 1 – {algo} | Skenario Fitness={sc} | "
              f"Pencarian Populasi (iter={FIXED_ITER})\n{sep}")
    global_log.append(header); print(header)

    results = {}

    for pop in POPULATION_SEARCH:
        exp_prefix = f"{sc_cfg['EXPERIMENT_NAME']}_pop{pop}_iter{FIXED_ITER}"

        existing = find_existing_exp(sc_cfg["RESULT_DIR"], exp_prefix)
        if existing:
            acc = read_best_scenario_acc(existing, bsk)
            results[pop] = acc
            msg = (f"  [SKIP] scen={sc} pop={pop:3d} iter={FIXED_ITER} "
                   f"→ Acc={acc:.6f} (sudah ada)")
            global_log.append(msg); print(msg)
            continue

        cfg = cfg_base.copy()
        cfg.update(sc_cfg)
        cfg["POP_SIZE"]        = pop
        cfg["MAX_ITER"]        = FIXED_ITER
        cfg["EXPERIMENT_NAME"] = exp_prefix

        try:
            acc, _ = run_one_experiment(cfg, X_feat, y, bsk)
            results[pop] = acc
            msg = f"  [OK]   scen={sc} pop={pop:3d} iter={FIXED_ITER} → Acc={acc:.6f}"
        except Exception as e:
            results[pop] = -1.0
            msg = f"  [FAIL] scen={sc} pop={pop:3d} iter={FIXED_ITER} → {e}"
            traceback.print_exc()

        global_log.append(msg); print(msg)

    best_pop = max(results, key=lambda p: results[p])
    best_acc = results[best_pop]

    summary = (f"\n  ★ BEST POP  scen={sc}: pop={best_pop} | Acc={best_acc:.6f}\n"
               f"  Ranking: {sorted(results.items(), key=lambda x: -x[1])}")
    global_log.append(summary); print(summary)

    _save_phase_table(
        sc_cfg["RESULT_DIR"], f"phase1_pop_search_scen{sc}.csv",
        results, "pop_size", f"acc_{bsk}", best_key=best_pop
    )

    return best_pop


# ================================================================
# FASE 2 – PENCARIAN ITERASI TERBAIK
# ================================================================
def phase2_search_iter(sc_cfg: dict, cfg_base: dict,
                       X_feat: np.ndarray, y: np.ndarray,
                       best_pop: int, global_log: list) -> int:
    algo = sc_cfg["META_ALGO"]
    sc   = sc_cfg["FITNESS_SCENARIO"]
    bsk  = sc_cfg["BEST_SCENARIO_KEY"]
    sep  = "─" * 70

    header = (f"\n{sep}\n"
              f"  FASE 2 – {algo} | Skenario Fitness={sc} | "
              f"Pencarian Iterasi (pop={best_pop})\n{sep}")
    global_log.append(header); print(header)

    results = {}

    for itr in ITER_SEARCH:
        exp_prefix = f"{sc_cfg['EXPERIMENT_NAME']}_pop{best_pop}_iter{itr}"

        existing = find_existing_exp(sc_cfg["RESULT_DIR"], exp_prefix)
        if existing:
            acc = read_best_scenario_acc(existing, bsk)
            results[itr] = acc
            msg = (f"  [SKIP] scen={sc} pop={best_pop} iter={itr:3d} "
                   f"→ Acc={acc:.6f} (sudah ada)")
            global_log.append(msg); print(msg)
            continue

        cfg = cfg_base.copy()
        cfg.update(sc_cfg)
        cfg["POP_SIZE"]        = best_pop
        cfg["MAX_ITER"]        = itr
        cfg["EXPERIMENT_NAME"] = exp_prefix

        try:
            acc, _ = run_one_experiment(cfg, X_feat, y, bsk)
            results[itr] = acc
            msg = f"  [OK]   scen={sc} pop={best_pop} iter={itr:3d} → Acc={acc:.6f}"
        except Exception as e:
            results[itr] = -1.0
            msg = f"  [FAIL] scen={sc} pop={best_pop} iter={itr:3d} → {e}"
            traceback.print_exc()

        global_log.append(msg); print(msg)

    best_iter = max(results, key=lambda i: results[i])
    best_acc  = results[best_iter]

    summary = (f"\n  ★ BEST ITER scen={sc}: iter={best_iter} (pop={best_pop}) "
               f"| Acc={best_acc:.6f}\n"
               f"  Ranking: {sorted(results.items(), key=lambda x: -x[1])}")
    global_log.append(summary); print(summary)

    _save_phase_table(
        sc_cfg["RESULT_DIR"], f"phase2_iter_search_scen{sc}.csv",
        results, "max_iter", f"acc_{bsk}", best_key=best_iter
    )

    return best_iter


# ================================================================
# MAIN
# ================================================================
def main():
    global_log  = []
    all_summary = {}   # sc → {best_pop, best_iter, best_acc, elapsed_mnt}
    t_global    = time.time()

    # ── Base config ──
    cfg_base = build_config()

    # ── CNN (dibangun sekali) ──
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
    print(f"[INIT] X_feat: {X_feat.shape} | Kelas: {len(set(y))}\n")

    scenario_configs = build_scenario_configs()

    # ── Header log ──
    header = (
        f"{'='*70}\n"
        f"  AUTO-RUN 2-FASE – CL_GWO | {len(FITNESS_SCENARIOS)} Skenario Fitness\n"
        f"  Skenario : {' → '.join(FITNESS_SCENARIOS)}\n"
        f"  Fase 1   : pop={POPULATION_SEARCH}  iter={FIXED_ITER} (tetap)\n"
        f"  Fase 2   : pop=best(Fase1)  iter={ITER_SEARCH}\n"
        f"{'='*70}"
    )
    global_log.append(header)
    print(header)

    # ── Loop tiap skenario fitness ──
    for sc_cfg in scenario_configs:
        sc   = sc_cfg["FITNESS_SCENARIO"]
        bsk  = sc_cfg["BEST_SCENARIO_KEY"]
        t_sc = time.time()

        print(f"\n{'#'*70}")
        print(f"  SKENARIO FITNESS = {sc}  |  eval-key: {bsk}")
        print(f"{'#'*70}")

        try:
            resume = load_resume(sc_cfg["RESULT_DIR"])

            if resume and resume.get("phase") == "done":
                best_pop  = resume["best_pop"]
                best_iter = resume["best_iter"]
                msg = (f"\n  [RESUME] scen={sc}: best_pop={best_pop} "
                       f"best_iter={best_iter} (sudah selesai sebelumnya)")
                global_log.append(msg); print(msg)

            else:
                # Fase 1
                if resume and resume.get("best_pop") and resume.get("phase") == "fase2":
                    best_pop = resume["best_pop"]
                    msg = f"\n  [RESUME] scen={sc}: Fase 1 sudah selesai → best_pop={best_pop}"
                    global_log.append(msg); print(msg)
                else:
                    best_pop = phase1_search_population(
                        sc_cfg, cfg_base, X_feat, y, global_log
                    )
                    save_resume(sc_cfg["RESULT_DIR"], "fase2", sc, best_pop=best_pop)

                # Fase 2
                best_iter = phase2_search_iter(
                    sc_cfg, cfg_base, X_feat, y, best_pop, global_log
                )
                save_resume(sc_cfg["RESULT_DIR"], "done", sc,
                            best_pop=best_pop, best_iter=best_iter)

            elapsed_m = (time.time() - t_sc) / 60
            best_exp_prefix = f"{sc_cfg['EXPERIMENT_NAME']}_pop{best_pop}_iter{best_iter}"
            best_exp_dir    = find_existing_exp(sc_cfg["RESULT_DIR"], best_exp_prefix)
            best_acc_final  = (read_best_scenario_acc(best_exp_dir, bsk)
                               if best_exp_dir else -1.0)

            all_summary[sc] = {
                "fitness_scenario":   sc,
                "eval_key":           bsk,
                "best_pop":           best_pop,
                "best_iter":          best_iter,
                "best_acc":           best_acc_final,
                "elapsed_mnt":        round(elapsed_m, 1),
            }

            row = (f"  ✓ scen={sc} | best_pop={best_pop:3d} | best_iter={best_iter:3d} "
                   f"| Acc={best_acc_final:.6f} | {elapsed_m:.1f} mnt")
            global_log.append(row); print(row)

        except Exception as e:
            msg = f"  ✗ GAGAL scen={sc}: {e}"
            global_log.append(msg); print(msg)
            traceback.print_exc()

    # ── Ringkasan akhir ──
    total_h = (time.time() - t_global) / 3600
    footer_lines = [
        f"\n{'='*70}",
        f"  SELESAI – CL_GWO All Scenarios | Total waktu: {total_h:.2f} jam",
        f"  {'SCEN':<6} {'EVAL KEY':<42} {'POP':>5} {'ITER':>6} {'ACC':>10}",
        f"  {'─'*70}",
    ]
    for sc, s in all_summary.items():
        footer_lines.append(
            f"  {sc:<6} {s['eval_key']:<42} {s['best_pop']:>5} "
            f"{s['best_iter']:>6} {s['best_acc']:>10.6f}"
        )
    footer_lines.append(f"{'='*70}")
    for ln in footer_lines:
        global_log.append(ln); print(ln)

    # Simpan summary JSON
    summary_json = os.path.join(BASE_DIR, "best_params_CLGWO_all_scenarios.json")
    with open(summary_json, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\n  Summary params → {summary_json}")

    # Simpan log teks
    log_path = os.path.join(BASE_DIR, "run_auto_CLGWO_all_scenarios.log")
    with open(log_path, "w") as f:
        f.write("\n".join(global_log))
    print(f"  Log lengkap    → {log_path}")


if __name__ == "__main__":
    main()
