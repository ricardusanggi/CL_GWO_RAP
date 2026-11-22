#!/usr/bin/env python
# ================================================================
# EKSPERIMEN 3: ANALISIS SENSITIVITAS HYPERPARAMETER CL_GWO
#
# Reviewer meminta grid search untuk dua hyperparameter utama:
#   - max_flag : [3, 5, 7, 9, 11]
#     (stagnasi counter, trigger refresh exemplar)
#   - alpha    : [0.1, 0.2, 0.3, 0.5]
#     (bobot penalti sparsity dalam fungsi objektif)
#
# Setup:
#   - Backbone: convnext_tiny (768-dim, sama dengan paper)
#   - Skenario G (seleksi murni), pop=25, iter=25
#   - Metrik: mean accuracy (5-Fold CV)
#
# Output:
#   - Tabel CSV hasil grid search
#   - Heatmap akurasi (max_flag × alpha)
#   - Line plot sensitivitas per parameter (marginal effect)
# ================================================================

import os
import sys
import csv
import time
import copy
import traceback

BASE_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "full_code_RAP")
sys.path.insert(0, BASE_MAIN)

RESULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from mealpy.utils.space import FloatVar
from mealpy.swarm_based.GWO import CL_GWO

from config import build_config
from utils.io import ensure_dir, now
from utils.logger import UnifiedLogger

from core.cnn_feature_extractor import CNNFeatureExtractor
from core.feature_constructor import FeatureConstructor
from core.fitness import Fitness
from core.objective import MetaObjective
from core.final_classification import run_classification


# ================================================================
# KONFIGURASI
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid hyperparameter
MAX_FLAG_LIST = [3, 5, 7, 9, 11]
ALPHA_LIST    = [0.1, 0.2, 0.3, 0.5]

# CL_GWO fixed params
POP_SIZE   = 25
MAX_ITER   = 25
W_MIN      = 0.2
W_MAX      = 2.0

FITNESS_SCENARIO = "G"


# ================================================================
# HELPER
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


def run_one(cfg, X_feat, y, max_flag, alpha):
    """Jalankan satu titik grid search."""
    ensure_dir(cfg["RESULT_DIR"])
    exp_name = f"sens_flag{max_flag}_alpha{str(alpha).replace('.','')}"
    exp_dir  = os.path.join(cfg["RESULT_DIR"], exp_name)
    ensure_dir(exp_dir)

    feat_dim    = X_feat.shape[1]
    constructor = FeatureConstructor(feat_dim)
    fitness     = Fitness(X_feat, y, cfg)

    problem = {
        "obj_func": MetaObjective(X_feat, fitness, constructor, cfg, alpha=alpha),
        "bounds":   FloatVar(lb=[0.0] * (2 * feat_dim), ub=[1.0] * (2 * feat_dim)),
        "minmax":   "min",
        "save_population": False,
    }

    optimizer = CL_GWO(
        epoch=MAX_ITER,
        pop_size=POP_SIZE,
        w_min=W_MIN,
        w_max=W_MAX,
        max_flag=max_flag
    )

    t0   = time.time()
    best = optimizer.solve(problem)
    elapsed = time.time() - t0

    decoded   = constructor.decode(best.solution)
    if decoded is None:
        return {"acc": 0.0, "n_sel": 0, "elapsed": elapsed}

    f_sel, f_dis = constructor.construct_features_only(X_feat, decoded)
    final_res = run_classification(
        X_feat=X_feat, y=y, decoded=decoded,
        f_sel=f_sel, f_dis=f_dis, cfg=cfg,
        mode=cfg["CLASSIFIER_MODE"].lower()
    )

    logger = UnifiedLogger(exp_dir, constructor, cfg)
    logger.save_final_classification_results(final_res)

    # Ambil akurasi Skenario G
    # final_res adalah dict: {"G_selected_only": {"mean_acc": ..., ...}, ...}
    acc = float(final_res.get("G_selected_only", {}).get("mean_acc", 0.0))

    return {
        "acc":     acc,
        "n_sel":   len(decoded["selected"]),
        "elapsed": elapsed,
    }


# ================================================================
# PLOT HASIL
# ================================================================
def plot_heatmap(grid, out_dir):
    """Heatmap akurasi: baris=max_flag, kolom=alpha."""
    ensure_dir(out_dir)
    data = np.array([[grid.get((mf, al), 0.0) for al in ALPHA_LIST]
                     for mf in MAX_FLAG_LIST])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        data,
        annot=True, fmt=".4f",
        xticklabels=[f"α={a}" for a in ALPHA_LIST],
        yticklabels=[f"max_flag={mf}" for mf in MAX_FLAG_LIST],
        cmap="YlOrRd", ax=ax,
        linewidths=0.5, linecolor="gray"
    )
    ax.set_title("Sensitivitas Hyperparameter CL_GWO\n(Akurasi Skenario G, 5-Fold CV)")
    ax.set_xlabel("Alpha (penalti sparsity)")
    ax.set_ylabel("max_flag (stagnasi counter)")
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Heatmap → {path}")


def plot_marginal(grid, out_dir):
    """Line plot: marginal effect tiap parameter."""
    ensure_dir(out_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Efek max_flag (rata-rata atas semua alpha)
    mf_accs = []
    for mf in MAX_FLAG_LIST:
        vals = [grid.get((mf, al), 0.0) for al in ALPHA_LIST]
        mf_accs.append(np.mean(vals))

    ax1.plot(MAX_FLAG_LIST, mf_accs, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.set_xlabel("max_flag")
    ax1.set_ylabel("Rata-rata Akurasi (atas semua α)")
    ax1.set_title("Sensitivitas max_flag")
    ax1.set_xticks(MAX_FLAG_LIST)
    ax1.grid(True, alpha=0.4)
    for x, y_val in zip(MAX_FLAG_LIST, mf_accs):
        ax1.annotate(f"{y_val:.4f}", (x, y_val), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    # Efek alpha (rata-rata atas semua max_flag)
    al_accs = []
    for al in ALPHA_LIST:
        vals = [grid.get((mf, al), 0.0) for mf in MAX_FLAG_LIST]
        al_accs.append(np.mean(vals))

    ax2.plot(ALPHA_LIST, al_accs, "s-", color="darkorange", linewidth=2, markersize=8)
    ax2.set_xlabel("Alpha (α)")
    ax2.set_ylabel("Rata-rata Akurasi (atas semua max_flag)")
    ax2.set_title("Sensitivitas Alpha (α)")
    ax2.set_xticks(ALPHA_LIST)
    ax2.grid(True, alpha=0.4)
    for x, y_val in zip(ALPHA_LIST, al_accs):
        ax2.annotate(f"{y_val:.4f}", (x, y_val), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    plt.suptitle("Analisis Sensitivitas Hyperparameter CL_GWO", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "marginal_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Marginal plot → {path}")


def save_csv(grid, out_dir):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "sensitivity_grid.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["max_flag", "alpha", "acc_scenario_G"])
        for mf in MAX_FLAG_LIST:
            for al in ALPHA_LIST:
                w.writerow([mf, al, f"{grid.get((mf, al), 0.0):.6f}"])
    print(f"[SAVED] Grid CSV → {path}")


# ================================================================
# RESUME STATE
# ================================================================
def load_done_set(out_dir):
    """Baca eksperimen yang sudah selesai dari CSV (untuk skip)."""
    csv_path = os.path.join(out_dir, "sensitivity_grid.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mf = int(row["max_flag"])
                al = float(row["alpha"])
                acc = float(row["acc_scenario_G"])
                if acc > 0:
                    done.add((mf, al))
    return done


# ================================================================
# MAIN
# ================================================================
def main():
    cfg_base = build_config()
    cfg_base["META_ALGO"]       = "CL_GWO"
    cfg_base["POP_SIZE"]        = POP_SIZE
    cfg_base["MAX_ITER"]        = MAX_ITER
    cfg_base["FITNESS_SCENARIO"] = FITNESS_SCENARIO
    cfg_base["CLASSIFIER_MODE"] = "ml"
    cfg_base["CLASSIFIER"]      = "svm"
    cfg_base["RESULT_DIR"]      = RESULT_BASE

    ensure_dir(RESULT_BASE)

    print("[INIT] Membangun CNN backbone ...")
    extractor = CNNFeatureExtractor(
        img_size=cfg_base["IMG_SIZE"],
        backbone=cfg_base["BACKBONE"],
        device=DEVICE
    )
    backbone_model, preprocess, feat_dim = extractor.build()

    print("[INIT] Memuat dataset ...")
    X_img, y = load_images(cfg_base["DATASET_DIR"], preprocess)
    print("[INIT] Ekstraksi fitur ...")
    X_feat = backbone_model.predict(X_img, batch_size=256)
    del X_img
    print(f"       X_feat: {X_feat.shape}")

    done_set = load_done_set(RESULT_BASE)
    grid     = {}

    # Muat hasil yang sudah ada
    if done_set:
        csv_path = os.path.join(RESULT_BASE, "sensitivity_grid.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mf  = int(row["max_flag"])
                al  = float(row["alpha"])
                acc = float(row["acc_scenario_G"])
                grid[(mf, al)] = acc

    total  = len(MAX_FLAG_LIST) * len(ALPHA_LIST)
    done_n = len(done_set)
    print(f"\n[INFO] Grid: {total} titik | Sudah selesai: {done_n}")

    for mf in MAX_FLAG_LIST:
        for al in ALPHA_LIST:
            key = (mf, al)
            if key in done_set:
                print(f"  [SKIP] max_flag={mf} alpha={al} → acc={grid[key]:.4f}")
                continue

            print(f"\n  Grid: max_flag={mf} | alpha={al}")
            cfg = copy.deepcopy(cfg_base)
            cfg["EXPERIMENT_NAME"] = f"sens_mf{mf}_a{str(al).replace('.','')}"

            try:
                res = run_one(cfg, X_feat, y, max_flag=mf, alpha=al)
                acc = res["acc"]
                print(f"  → acc={acc:.4f} | n_sel={res['n_sel']} | {res['elapsed']:.0f}s")
            except Exception as e:
                acc = 0.0
                print(f"  [FAIL]: {e}")
                traceback.print_exc()

            grid[key] = acc
            # Simpan progres setelah tiap titik
            save_csv(grid, RESULT_BASE)

    # Visualisasi akhir
    plot_heatmap(grid, RESULT_BASE)
    plot_marginal(grid, RESULT_BASE)

    # Cetak tabel ringkasan
    print(f"\n{'='*55}")
    print(f"  {'max_flag':>8} {'alpha':>6}  {'acc':>10}")
    print(f"  {'─'*42}")
    for mf in MAX_FLAG_LIST:
        for al in ALPHA_LIST:
            mark = " "
            if grid.get((mf, al), 0.0) == max(grid.values()):
                mark = "★"
            print(f"{mark} {mf:>8} {al:>6.1f}  {grid.get((mf,al),0.0):>10.4f}")
    print(f"{'='*55}")

    best_key = max(grid, key=lambda k: grid[k])
    print(f"\n★ BEST: max_flag={best_key[0]}, alpha={best_key[1]}, acc={grid[best_key]:.4f}")
    print("\n[DONE] Eksperimen 3 selesai.")


if __name__ == "__main__":
    main()
