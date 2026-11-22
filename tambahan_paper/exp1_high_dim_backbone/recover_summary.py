#!/usr/bin/env python
# Baca ulang hasil dari final_classification_results.csv yang sudah ada,
# lalu rebuild summary CSV dan plot.
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "full_code_RAP"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.io import ensure_dir

RESULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCENARIOS   = ["G", "A", "E"]

KEY_MAP = {
    "G": "G_selected_only",
    "A": "A_selected_plus_f_selected",
    "E": "E_all_plus_f_selected_plus_f_discarded",
}

HIGH_DIM_BACKBONES = [
    ("convnext_base",   1024),
    ("convnext_large",  1536),
    ("resnet50",        2048),
    ("efficientnet_b5", 2048),
    ("densenet161",     2208),
]

summary_rows = []

for backbone_name, feat_dim in HIGH_DIM_BACKBONES:
    for scenario in SCENARIOS:
        sc_dir = os.path.join(RESULT_BASE, f"backbone_{backbone_name}", f"scenario_{scenario}")
        if not os.path.isdir(sc_dir):
            print(f"  [MISS] {backbone_name} Sc{scenario}: folder tidak ada")
            summary_rows.append({"backbone": backbone_name, "dim": feat_dim,
                                  "scenario": scenario, "acc": 0.0, "n_sel": 0, "elapsed": 0})
            continue

        # Cari final_classification_results.csv
        clf_csv = None
        for root, dirs, files in os.walk(sc_dir):
            for f in files:
                if f == "final_classification_results.csv":
                    clf_csv = os.path.join(root, f)
                    break
            if clf_csv:
                break

        if clf_csv is None:
            print(f"  [MISS] {backbone_name} Sc{scenario}: tidak ada final_classification_results.csv")
            summary_rows.append({"backbone": backbone_name, "dim": feat_dim,
                                  "scenario": scenario, "acc": 0.0, "n_sel": 0, "elapsed": 0})
            continue

        # Baca akurasi
        key    = KEY_MAP[scenario]
        acc    = 0.0
        n_sel  = 0
        with open(clf_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["scenario"] == key:
                    acc   = float(row["mean_accuracy"])
                    n_sel = int(row["num_features"])
                    break

        print(f"  {backbone_name} Sc{scenario}: acc={acc:.4f} n_sel={n_sel}")
        summary_rows.append({"backbone": backbone_name, "dim": feat_dim,
                              "scenario": scenario, "acc": acc, "n_sel": n_sel, "elapsed": 0})

# Simpan CSV
csv_path = os.path.join(RESULT_BASE, "summary_high_dim.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["backbone", "dim", "scenario", "acc", "n_selected", "elapsed_sec"])
    for r in summary_rows:
        w.writerow([r["backbone"], r["dim"], r["scenario"],
                    f"{r['acc']:.6f}", r["n_sel"], r["elapsed"]])
print(f"\n[SAVED] {csv_path}")

# Plot
backbones = [b for b, _ in HIGH_DIM_BACKBONES]
colors    = {"G": "steelblue", "A": "darkorange", "E": "green"}
fig, ax   = plt.subplots(figsize=(max(8, len(backbones) * 2), 5))
x     = np.arange(len(backbones))
width = 0.25

for i, sc in enumerate(SCENARIOS):
    accs = []
    for bb in backbones:
        rows = [r for r in summary_rows if r["backbone"] == bb and r["scenario"] == sc]
        accs.append(rows[0]["acc"] if rows else 0.0)
    ax.bar(x + i * width, accs, width, label=f"Scenario {sc}", color=colors[sc], alpha=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels(backbones, rotation=20, ha="right")
ax.set_ylabel("Akurasi")
ax.set_title("CL_GWO: Akurasi per Backbone (dim >768) × Skenario")
ax.legend(); ax.set_ylim(0, 1.05)
plt.tight_layout()

plot_path = os.path.join(RESULT_BASE, "comparison_high_dim.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] {plot_path}")
print("\n[DONE] Recovery selesai.")
