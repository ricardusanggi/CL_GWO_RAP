#!/usr/bin/env python
# Baca ulang hasil dari final_classification_results.csv yang sudah ada,
# lalu rebuild grid CSV dan plot.
import os, sys, csv, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "full_code_RAP"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io import ensure_dir

RESULT_BASE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAX_FLAG_LIST = [3, 5, 7, 9, 11]
ALPHA_LIST    = [0.1, 0.2, 0.3, 0.5]

grid = {}

for mf in MAX_FLAG_LIST:
    for al in ALPHA_LIST:
        # Nama folder: sens_flag{mf}_alpha{al_no_dot}
        al_str   = str(al).replace(".", "")
        exp_name = f"sens_flag{mf}_alpha{al_str}"
        exp_dir  = os.path.join(RESULT_BASE, exp_name)

        if not os.path.isdir(exp_dir):
            print(f"  [MISS] {exp_name}")
            grid[(mf, al)] = 0.0
            continue

        clf_csv = os.path.join(exp_dir, "final_classification_results.csv")
        if not os.path.exists(clf_csv):
            print(f"  [MISS] {exp_name}: tidak ada CSV")
            grid[(mf, al)] = 0.0
            continue

        acc = 0.0
        with open(clf_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["scenario"] == "G_selected_only":
                    acc = float(row["mean_accuracy"])
                    break

        print(f"  max_flag={mf} alpha={al}: acc={acc:.4f}")
        grid[(mf, al)] = acc

# Simpan CSV
csv_path = os.path.join(RESULT_BASE, "sensitivity_grid.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["max_flag", "alpha", "acc_scenario_G"])
    for mf in MAX_FLAG_LIST:
        for al in ALPHA_LIST:
            w.writerow([mf, al, f"{grid.get((mf, al), 0.0):.6f}"])
print(f"\n[SAVED] {csv_path}")

# Heatmap
data = np.array([[grid.get((mf, al), 0.0) for al in ALPHA_LIST]
                 for mf in MAX_FLAG_LIST])

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(data, annot=True, fmt=".4f",
            xticklabels=[f"α={a}" for a in ALPHA_LIST],
            yticklabels=[f"max_flag={mf}" for mf in MAX_FLAG_LIST],
            cmap="YlOrRd", ax=ax, linewidths=0.5, linecolor="gray")
ax.set_title("Sensitivitas Hyperparameter CL_GWO\n(Akurasi Skenario G, 5-Fold CV)")
ax.set_xlabel("Alpha (penalti sparsity)")
ax.set_ylabel("max_flag (stagnasi counter)")
plt.tight_layout()
fig.savefig(os.path.join(RESULT_BASE, "heatmap_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] heatmap")

# Marginal plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
mf_accs = [np.mean([grid.get((mf, al), 0.0) for al in ALPHA_LIST]) for mf in MAX_FLAG_LIST]
al_accs = [np.mean([grid.get((mf, al), 0.0) for mf in MAX_FLAG_LIST]) for al in ALPHA_LIST]

ax1.plot(MAX_FLAG_LIST, mf_accs, "o-", color="steelblue", linewidth=2, markersize=8)
ax1.set_xlabel("max_flag"); ax1.set_ylabel("Rata-rata Akurasi")
ax1.set_title("Sensitivitas max_flag"); ax1.set_xticks(MAX_FLAG_LIST); ax1.grid(True, alpha=0.4)
for x, y in zip(MAX_FLAG_LIST, mf_accs):
    ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

ax2.plot(ALPHA_LIST, al_accs, "s-", color="darkorange", linewidth=2, markersize=8)
ax2.set_xlabel("Alpha (α)"); ax2.set_ylabel("Rata-rata Akurasi")
ax2.set_title("Sensitivitas Alpha (α)"); ax2.set_xticks(ALPHA_LIST); ax2.grid(True, alpha=0.4)
for x, y in zip(ALPHA_LIST, al_accs):
    ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

plt.suptitle("Analisis Sensitivitas Hyperparameter CL_GWO", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(RESULT_BASE, "marginal_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[SAVED] marginal plot")

# Print ringkasan
print(f"\n{'='*55}")
print(f"  {'max_flag':>8} {'alpha':>6}  {'acc':>10}")
print(f"  {'─'*42}")
best_key = max(grid, key=lambda k: grid[k])
for mf in MAX_FLAG_LIST:
    for al in ALPHA_LIST:
        mark = "★" if (mf, al) == best_key else " "
        print(f"{mark} {mf:>8} {al:>6.1f}  {grid.get((mf,al),0.0):>10.4f}")
print(f"\n★ BEST: max_flag={best_key[0]}, alpha={best_key[1]}, acc={grid[best_key]:.4f}")
print("\n[DONE] Recovery selesai.")
