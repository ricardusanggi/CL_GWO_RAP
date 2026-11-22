#!/usr/bin/env python
# ================================================================
# EKSPERIMEN 1: BACKBONE DIMENSI TINGGI + ANALISIS FEATURE CONSTRUCTION
#
# Reviewer meminta bukti bahwa Feature Construction berguna pada
# kondisi backbone yang menghasilkan fitur > 768-dim.
#
# Backbone yang diuji (semua > 768-dim):
#   - resnet50       (2048-dim)
#   - densenet161    (2208-dim)
#   - efficientnet_b5 (2048-dim)
#   - convnext_base  (1024-dim)
#   - convnext_large (1536-dim)
#
# Untuk tiap backbone, jalankan CL_GWO dengan skenario:
#   - Scenario G  : X_sel saja (seleksi murni, tanpa FC)
#   - Scenario A  : X_sel + f_sel (seleksi + FC dari selected)
#   - Scenario E  : X_all + f_sel + f_dis (full + FC, skenario terburuk)
#
# Juga menghasilkan analisis distribusi fitur sebelum/sesudah FC
# untuk Skenario E (mengapa E paling buruk).
# ================================================================

import os
import sys
import csv
import time
import copy
import traceback

# Path ke kode utama
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
# KONFIGURASI
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Backbone > 768-dim
HIGH_DIM_BACKBONES = [
    ("convnext_base",   1024),
    ("convnext_large",  1536),
    ("resnet50",        2048),
    ("efficientnet_b5", 2048),
    ("densenet161",     2208),
]

# Skenario yang dibandingkan per backbone
SCENARIOS = ["G", "A", "E"]

# CL_GWO dengan setting terbaik yang sudah ditemukan (dari main_auto_all_methods)
# Gunakan nilai konservatif jika belum punya best params
POP_SIZE = 25
MAX_ITER = 25
ALPHA    = 0.3


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


def run_one(cfg, X_feat, y, scenario_label):
    """Jalankan satu eksperimen CL_GWO pada fitur X_feat dengan skenario tertentu."""
    ensure_dir(cfg["RESULT_DIR"])
    exp_name = f"{cfg['EXPERIMENT_NAME']}_{now()}"
    exp_dir  = os.path.join(cfg["RESULT_DIR"], exp_name)
    ensure_dir(exp_dir)

    feat_dim    = X_feat.shape[1]
    constructor = FeatureConstructor(feat_dim)
    fitness     = Fitness(X_feat, y, cfg)

    problem = {
        "obj_func": MetaObjective(X_feat, fitness, constructor, cfg, alpha=ALPHA),
        "bounds":   FloatVar(lb=[0.0] * (2 * feat_dim), ub=[1.0] * (2 * feat_dim)),
        "minmax":   "min",
        "save_population": False,
    }

    optimizer = create_optimizer(cfg)

    t0   = time.time()
    best = optimizer.solve(problem)
    elapsed = time.time() - t0

    decoded   = constructor.decode(best.solution)
    f_sel, f_dis = constructor.construct_features_only(X_feat, decoded)
    final_res = run_classification(
        X_feat=X_feat, y=y, decoded=decoded,
        f_sel=f_sel, f_dis=f_dis, cfg=cfg,
        mode=cfg["CLASSIFIER_MODE"].lower()
    )

    logger = UnifiedLogger(exp_dir, constructor, cfg)
    logger.save_final_classification_results(final_res)

    # Ambil akurasi skenario yang diminta
    # final_res adalah dict: {"G_selected_only": {"mean_acc": ..., ...}, ...}
    key_map = {
        "G": "G_selected_only",
        "A": "A_selected_plus_f_selected",
        "E": "E_all_plus_f_selected_plus_f_discarded",
    }
    key = key_map.get(scenario_label, "G_selected_only")
    acc = float(final_res.get(key, {}).get("mean_acc", 0.0))

    n_sel = len(decoded["selected"])
    return {
        "acc":      acc,
        "n_sel":    n_sel,
        "elapsed":  elapsed,
        "exp_dir":  exp_dir,
        "decoded":  decoded,
    }


# ================================================================
# ANALISIS DISTRIBUSI FITUR (untuk memahami mengapa Skenario E buruk)
# ================================================================
def analyze_feature_distribution(X_feat, decoded, constructor, backbone_name, out_dir):
    """
    Visualisasi distribusi fitur sebelum dan sesudah Feature Construction.
    Khusus untuk memahami Skenario E (mengapa penambahan f_sel & f_dis ke X_all bisa buruk).
    """
    ensure_dir(out_dir)

    f_sel, f_dis = constructor.construct_features_only(X_feat, decoded)

    # Statistik distribusi
    stats = {}
    stats["X_all_mean"]   = float(np.mean(np.abs(X_feat)))
    stats["X_all_std"]    = float(np.std(X_feat))
    stats["X_all_range"]  = float(X_feat.max() - X_feat.min())

    stats["f_sel_mean"]   = float(np.mean(np.abs(f_sel)))
    stats["f_sel_std"]    = float(np.std(f_sel))
    stats["f_sel_range"]  = float(f_sel.max() - f_sel.min())

    stats["f_dis_mean"]   = float(np.mean(np.abs(f_dis)))
    stats["f_dis_std"]    = float(np.std(f_dis))
    stats["f_dis_range"]  = float(f_dis.max() - f_dis.min())

    # Plot 1: Distribusi nilai fitur (histogram per grup)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Distribusi Fitur – {backbone_name} (Skenario E Analysis)", fontsize=12)

    for ax, data, label in zip(
        axes,
        [X_feat.flatten(), f_sel.flatten(), f_dis.flatten()],
        ["X_all (fitur asli)", "f_sel (konstruksi selected)", "f_dis (konstruksi discarded)"]
    ):
        ax.hist(np.clip(data, -100, 100), bins=80, color="steelblue", alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel("Nilai")
        ax.set_ylabel("Frekuensi")
        ax.set_xlim(-100, 100)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dist_{backbone_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Per-fitur variance (X_all vs f_sel vs f_dis)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    vars_all = np.var(X_feat, axis=0)
    bars = ax2.bar(
        ["X_all\n(tiap fitur)", "f_sel", "f_dis"],
        [float(np.mean(vars_all)), float(np.var(f_sel)), float(np.var(f_dis))],
        color=["steelblue", "darkorange", "green"]
    )
    ax2.set_title(f"Rata-rata Varians per Grup – {backbone_name}")
    ax2.set_ylabel("Varians")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"variance_{backbone_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Simpan statistik ke CSV
    csv_path = os.path.join(out_dir, f"feature_stats_{backbone_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, v])

    print(f"  [DIST] {backbone_name}: X_all_std={stats['X_all_std']:.4f} "
          f"| f_sel_std={stats['f_sel_std']:.4f} | f_dis_std={stats['f_dis_std']:.4f}")
    return stats


# ================================================================
# RINGKASAN
# ================================================================
def save_summary(summary_rows, out_dir):
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "summary_high_dim.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "dim", "scenario", "acc", "n_selected", "elapsed_sec"])
        for r in summary_rows:
            w.writerow([r["backbone"], r["dim"], r["scenario"],
                        f"{r['acc']:.6f}", r["n_sel"], f"{r['elapsed']:.1f}"])
    print(f"\n[SAVED] Summary → {csv_path}")

    # Plot perbandingan akurasi per backbone × skenario
    backbones = sorted(set(r["backbone"] for r in summary_rows))
    scenarios = SCENARIOS
    colors    = {"G": "steelblue", "A": "darkorange", "E": "green"}

    fig, ax = plt.subplots(figsize=(max(8, len(backbones) * 2), 5))
    x       = np.arange(len(backbones))
    width   = 0.25

    for i, sc in enumerate(scenarios):
        accs = []
        for bb in backbones:
            rows = [r for r in summary_rows if r["backbone"] == bb and r["scenario"] == sc]
            accs.append(rows[0]["acc"] if rows else 0.0)
        ax.bar(x + i * width, accs, width, label=f"Scenario {sc}", color=colors[sc], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(backbones, rotation=20, ha="right")
    ax.set_ylabel("Akurasi")
    ax.set_title("CL_GWO: Akurasi per Backbone (dim >768) × Skenario")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "comparison_high_dim.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot    → {plot_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    cfg_base = build_config()
    cfg_base["META_ALGO"]        = "CL_GWO"
    cfg_base["POP_SIZE"]         = POP_SIZE
    cfg_base["MAX_ITER"]         = MAX_ITER
    cfg_base["CLASSIFIER_MODE"]  = "ml"
    cfg_base["CLASSIFIER"]       = "svm"

    ensure_dir(RESULT_BASE)
    summary_rows = []

    for backbone_name, feat_dim_expected in HIGH_DIM_BACKBONES:
        print(f"\n{'='*70}")
        print(f"  BACKBONE: {backbone_name} (dim={feat_dim_expected})")
        print(f"{'='*70}")

        # Ekstrak fitur CNN
        extractor = CNNFeatureExtractor(
            img_size=cfg_base["IMG_SIZE"],
            backbone=backbone_name,
            device=DEVICE
        )
        backbone_model, preprocess, feat_dim = extractor.build()
        print(f"  Memuat dataset dari {cfg_base['DATASET_DIR']}")
        X_img, y = load_images(cfg_base["DATASET_DIR"], preprocess)
        print(f"  Ekstraksi fitur CNN ...")
        X_feat = backbone_model.predict(X_img, batch_size=256)
        del X_img
        print(f"  X_feat: {X_feat.shape}")

        bb_result_dir  = os.path.join(RESULT_BASE, f"backbone_{backbone_name}")
        dist_out_dir   = os.path.join(bb_result_dir, "feature_dist_analysis")

        for scenario in SCENARIOS:
            print(f"\n  Skenario {scenario} ...")
            cfg = copy.deepcopy(cfg_base)
            cfg["BACKBONE"]          = backbone_name
            cfg["FITNESS_SCENARIO"]  = scenario
            cfg["RESULT_DIR"]        = os.path.join(bb_result_dir, f"scenario_{scenario}")
            cfg["EXPERIMENT_NAME"]   = f"clgwo_{backbone_name}_sc{scenario}"

            try:
                res = run_one(cfg, X_feat, y, scenario)
                print(f"  → acc={res['acc']:.4f} | n_sel={res['n_sel']} | {res['elapsed']:.0f}s")

                # Analisis distribusi untuk Skenario E
                if scenario == "E":
                    constructor = FeatureConstructor(feat_dim)
                    analyze_feature_distribution(
                        X_feat, res["decoded"], constructor, backbone_name, dist_out_dir
                    )

                summary_rows.append({
                    "backbone": backbone_name,
                    "dim":      feat_dim,
                    "scenario": scenario,
                    "acc":      res["acc"],
                    "n_sel":    res["n_sel"],
                    "elapsed":  res["elapsed"],
                })
            except Exception as e:
                print(f"  [FAIL] {backbone_name} Sc{scenario}: {e}")
                traceback.print_exc()
                summary_rows.append({
                    "backbone": backbone_name, "dim": feat_dim,
                    "scenario": scenario, "acc": 0.0,
                    "n_sel": 0, "elapsed": 0,
                })

        del X_feat

    save_summary(summary_rows, RESULT_BASE)
    print("\n[DONE] Eksperimen 1 selesai.")


if __name__ == "__main__":
    main()
