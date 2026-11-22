#!/usr/bin/env python
# ================================================================
# EKSPERIMEN 2: BASELINE NON-METAHEURISTIK
#
# Reviewer meminta perbandingan dengan baseline selain metaheuristik.
# Metode yang diimplementasikan:
#   1. LASSO       – L1-regularization feature selection
#   2. mRMR        – Minimum Redundancy Maximum Relevance
#   3. CNN-Attn    – CNN head dengan attention mechanism untuk
#                    automatic feature weighting (soft selection)
#
# Setup eksperimen sama dengan main pipeline:
#   - Backbone: convnext_tiny (768-dim, sama dengan paper utama)
#   - 5-Fold Stratified CV
#   - Classifier: SVM (sama dengan final_classification di paper)
#   - Metrik: mean accuracy ± std
#
# Pembanding utama: CL_GWO pada Skenario G (seleksi murni)
# ================================================================

import os
import sys
import csv
import time
import warnings
warnings.filterwarnings("ignore")

BASE_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "full_code_RAP")
sys.path.insert(0, BASE_MAIN)

RESULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from config import build_config
from utils.io import ensure_dir, now
from core.cnn_feature_extractor import CNNFeatureExtractor


# ================================================================
# DEVICE
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

CV_FOLDS     = 5
RANDOM_STATE = 42


# ================================================================
# LOAD IMAGES
# ================================================================
def load_images(dataset_dir, preprocess):
    X, y_raw = [], []
    for label in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            try:
                img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
                img = preprocess(img)
                X.append(img.numpy().transpose(1, 2, 0))
                y_raw.append(label)
            except Exception:
                continue
    return np.asarray(X), LabelEncoder().fit_transform(y_raw)


# ================================================================
# 1. LASSO FEATURE SELECTION
#    Gunakan LassoCV untuk mendapatkan koefisien non-zero,
#    lalu evaluasi fitur terpilih dengan SVM di dalam K-Fold.
# ================================================================
def run_lasso(X_feat, y):
    """
    Feature selection via LASSO (L1 Logistic Regression).
    Karena LASSO untuk klasifikasi multi-kelas menggunakan
    LogisticRegression(penalty='l1'), bukan LassoCV regresi.
    Threshold koefisien: fitur dengan mean(|coef|) > 0 pada semua kelas.
    """
    print("\n[LASSO] Memulai evaluasi ...")
    kf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    n_sels = []

    for fold_i, (tr, te) in enumerate(kf.split(X_feat, y)):
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]

        scaler  = StandardScaler()
        Xtr_sc  = scaler.fit_transform(Xtr)
        Xte_sc  = scaler.transform(Xte)

        # L1 logistic untuk feature selection
        lasso_sel = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            max_iter=2000, random_state=RANDOM_STATE, multi_class="ovr"
        )
        lasso_sel.fit(Xtr_sc, ytr)

        # Pilih fitur dengan coef non-zero (rata-rata absolute > threshold)
        coef_mean = np.mean(np.abs(lasso_sel.coef_), axis=0)
        threshold = 1e-4
        selected  = np.where(coef_mean > threshold)[0]

        if len(selected) == 0:
            # Fallback: ambil top-10% fitur
            selected = np.argsort(coef_mean)[-max(1, int(0.1 * len(coef_mean))):]

        Xtr_sel = Xtr_sc[:, selected]
        Xte_sel = Xte_sc[:, selected]

        # Klasifikasi dengan SVM
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
        clf.fit(Xtr_sel, ytr)
        acc = accuracy_score(yte, clf.predict(Xte_sel))
        scores.append(acc)
        n_sels.append(len(selected))
        print(f"  Fold {fold_i+1}: acc={acc:.4f} | n_sel={len(selected)}")

    return {
        "method": "LASSO",
        "mean_acc": float(np.mean(scores)),
        "std_acc":  float(np.std(scores)),
        "mean_n_sel": float(np.mean(n_sels)),
        "scores": scores,
    }


# ================================================================
# 2. mRMR FEATURE SELECTION
#    Implementasi mRMR dari scratch menggunakan mutual information.
#    sklearn.feature_selection.mutual_info_classif digunakan
#    sebagai estimator MI untuk menghindari dependensi paket tambahan.
# ================================================================
def mrmr_select(X, y, n_features_to_select):
    """
    mRMR: Minimum Redundancy Maximum Relevance.
    Greedy selection: pilih fitur satu per satu yang memaksimalkan:
        score(i) = MI(X_i, y) - (1/|S|) * sum_{j in S} MI(X_i, X_j)
    """
    from sklearn.feature_selection import mutual_info_classif

    n_total = X.shape[1]
    n_sel   = min(n_features_to_select, n_total)

    # Relevansi: MI tiap fitur dengan label
    relevance = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    # Redundansi antar fitur (MI pairwise) – mahal untuk dim besar,
    # komputasi lazy: hitung saat dibutuhkan
    selected = []
    remaining = list(range(n_total))

    # Pilih fitur pertama dengan relevansi tertinggi
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_sel and remaining:
        best_score = -np.inf
        best_feat  = None

        for feat in remaining:
            rel = relevance[feat]
            # Rata-rata redundansi terhadap fitur yang sudah dipilih
            mi_with_selected = []
            for s in selected:
                # Pakai korelasi sebagai proxy MI (cepat)
                corr = abs(np.corrcoef(X[:, feat], X[:, s])[0, 1])
                mi_with_selected.append(corr)

            redundancy = np.mean(mi_with_selected) if mi_with_selected else 0.0
            score = rel - redundancy

            if score > best_score:
                best_score = score
                best_feat  = feat

        selected.append(best_feat)
        remaining.remove(best_feat)

    return selected


def run_mrmr(X_feat, y, n_select_ratio=0.3):
    """
    Evaluasi mRMR dengan SVM di dalam K-Fold.
    n_select_ratio: proporsi fitur yang dipilih dari total.
    """
    print("\n[mRMR] Memulai evaluasi ...")
    kf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores  = []
    n_sels  = []
    n_to_sel = max(1, int(n_select_ratio * X_feat.shape[1]))

    for fold_i, (tr, te) in enumerate(kf.split(X_feat, y)):
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(Xtr)
        Xte_sc = scaler.transform(Xte)

        t0       = time.time()
        selected = mrmr_select(Xtr_sc, ytr, n_to_sel)
        t_sel    = time.time() - t0

        Xtr_sel = Xtr_sc[:, selected]
        Xte_sel = Xte_sc[:, selected]

        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
        clf.fit(Xtr_sel, ytr)
        acc = accuracy_score(yte, clf.predict(Xte_sel))
        scores.append(acc)
        n_sels.append(len(selected))
        print(f"  Fold {fold_i+1}: acc={acc:.4f} | n_sel={len(selected)} | sel_time={t_sel:.1f}s")

    return {
        "method": "mRMR",
        "mean_acc": float(np.mean(scores)),
        "std_acc":  float(np.std(scores)),
        "mean_n_sel": float(np.mean(n_sels)),
        "scores": scores,
    }


# ================================================================
# 3. CNN + ATTENTION (Automatic Feature Weighting)
#    MLP head dengan Channel Attention (SE-block style):
#    Squeeze-and-Excitation attention over fitur backbone,
#    soft feature selection via learned weights.
# ================================================================
class AttentionHead(nn.Module):
    """
    MLP classifier dengan Channel Attention (SE-style) untuk
    automatic feature weighting/selection.

    Arsitektur:
        input [N, D]
        → SE Attention: FC(D→D//r) → ReLU → FC(D//r→D) → Sigmoid → gate
        → gated_input = input * gate
        → FC(D→hidden) → BN → ReLU → Dropout
        → FC(hidden→n_classes)
    """
    def __init__(self, in_dim, n_classes, hidden=256, reduction=16, dropout=0.3):
        super().__init__()
        r = max(1, in_dim // reduction)

        # Channel Attention (SE block)
        self.se = nn.Sequential(
            nn.Linear(in_dim, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, in_dim, bias=False),
            nn.Sigmoid()
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        gate    = self.se(x)
        x_gated = x * gate
        return self.classifier(x_gated), gate


def run_cnn_attention(X_feat, y, epochs=50, batch_size=32, lr=1e-3):
    """
    Evaluasi CNN-Attention dalam K-Fold.
    Model di-train dari scratch tiap fold.
    Gate attention di akhir digunakan sebagai ukuran feature importance.
    """
    print("\n[CNN-Attn] Memulai evaluasi ...")
    kf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores  = []
    n_classes = len(np.unique(y))

    for fold_i, (tr, te) in enumerate(kf.split(X_feat, y)):
        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(Xtr).astype(np.float32)
        Xte_sc = scaler.transform(Xte).astype(np.float32)

        in_dim = Xtr_sc.shape[1]
        model  = AttentionHead(in_dim, n_classes).to(DEVICE)
        opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Mini-batch training
        X_t = torch.from_numpy(Xtr_sc).to(DEVICE)
        y_t = torch.from_numpy(ytr).long().to(DEVICE)

        model.train()
        for ep in range(epochs):
            perm = torch.randperm(len(X_t))
            for start in range(0, len(X_t), batch_size):
                idx    = perm[start:start + batch_size]
                logits, _ = model(X_t[idx])
                loss   = loss_fn(logits, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Evaluasi
        model.eval()
        with torch.no_grad():
            X_te_t  = torch.from_numpy(Xte_sc).to(DEVICE)
            logits, gate = model(X_te_t)
            preds   = logits.argmax(dim=1).cpu().numpy()
            acc     = accuracy_score(yte, preds)
            avg_gate = gate.mean(dim=0).cpu().numpy()

        scores.append(acc)
        n_effective = int((avg_gate > 0.5).sum())
        print(f"  Fold {fold_i+1}: acc={acc:.4f} | eff_features(gate>0.5)={n_effective}")

        del model
        torch.cuda.empty_cache()

    return {
        "method": "CNN-Attention",
        "mean_acc": float(np.mean(scores)),
        "std_acc":  float(np.std(scores)),
        "mean_n_sel": "N/A (soft)",
        "scores": scores,
    }


# ================================================================
# SIMPAN HASIL
# ================================================================
def save_results(results, out_dir, clgwo_acc=None, clgwo_std=None):
    ensure_dir(out_dir)

    # CSV
    csv_path = os.path.join(out_dir, "baseline_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_acc", "std_acc", "mean_n_sel"])
        for r in results:
            w.writerow([r["method"], f"{r['mean_acc']:.6f}",
                        f"{r['std_acc']:.6f}", r["mean_n_sel"]])
        if clgwo_acc is not None:
            w.writerow(["CL_GWO (Scenario G)", f"{clgwo_acc:.6f}",
                        f"{clgwo_std:.6f}" if clgwo_std else "N/A", "variable"])
    print(f"\n[SAVED] CSV → {csv_path}")

    # Bar plot
    methods = [r["method"] for r in results]
    accs    = [r["mean_acc"] for r in results]
    stds    = [r["std_acc"] for r in results]
    colors  = ["steelblue", "darkorange", "green"]

    if clgwo_acc is not None:
        methods.append("CL_GWO\n(Scenario G)")
        accs.append(clgwo_acc)
        stds.append(clgwo_std if clgwo_std else 0.0)
        colors.append("crimson")

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2), 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color=colors[:len(methods)], alpha=0.85)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Akurasi (5-Fold CV)")
    ax.set_title("Perbandingan: CL_GWO vs Baseline Non-Metaheuristik")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=clgwo_acc if clgwo_acc else 0, color="crimson", linestyle="--", alpha=0.5,
               label="CL_GWO reference")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "baseline_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot → {plot_path}")

    # Print tabel ringkasan
    print("\n" + "=" * 60)
    print(f"  {'Metode':<25} {'Akurasi':>10} {'±Std':>8} {'n_sel':>8}")
    print("  " + "─" * 55)
    for r in results:
        print(f"  {r['method']:<25} {r['mean_acc']:>10.4f} {r['std_acc']:>8.4f} "
              f"{str(r['mean_n_sel']):>8}")
    if clgwo_acc is not None:
        print(f"  {'CL_GWO (Sc. G)':<25} {clgwo_acc:>10.4f} "
              f"{(clgwo_std if clgwo_std else 0.0):>8.4f} {'variable':>8}")
    print("=" * 60)


# ================================================================
# MAIN
# ================================================================
def main():
    cfg = build_config()

    print("[INIT] Membangun CNN backbone (convnext_tiny, 768-dim) ...")
    extractor = CNNFeatureExtractor(
        img_size=cfg["IMG_SIZE"],
        backbone=cfg["BACKBONE"],   # convnext_tiny dari config
        device=DEVICE
    )
    backbone_model, preprocess, feat_dim = extractor.build()
    print(f"       Feature dim: {feat_dim}")

    print("[INIT] Memuat dataset ...")
    X_img, y = load_images(cfg["DATASET_DIR"], preprocess)

    print("[INIT] Ekstraksi fitur ...")
    X_feat = backbone_model.predict(X_img, batch_size=256)
    del X_img
    print(f"       X_feat: {X_feat.shape}")

    ensure_dir(RESULT_BASE)
    all_results = []

    # Jalankan tiap metode baseline
    t0 = time.time()
    all_results.append(run_lasso(X_feat, y))
    print(f"  LASSO selesai ({(time.time()-t0)/60:.1f} mnt)")

    t0 = time.time()
    all_results.append(run_mrmr(X_feat, y, n_select_ratio=0.3))
    print(f"  mRMR selesai ({(time.time()-t0)/60:.1f} mnt)")

    t0 = time.time()
    all_results.append(run_cnn_attention(X_feat, y))
    print(f"  CNN-Attention selesai ({(time.time()-t0)/60:.1f} mnt)")

    # Nilai CL_GWO dari paper (isi manual setelah eksperimen utama selesai)
    # Set None jika belum ada
    CL_GWO_BEST_ACC = None   # contoh: 0.9342
    CL_GWO_BEST_STD = None   # contoh: 0.0121

    save_results(all_results, RESULT_BASE,
                 clgwo_acc=CL_GWO_BEST_ACC, clgwo_std=CL_GWO_BEST_STD)
    print("\n[DONE] Eksperimen 2 selesai.")


if __name__ == "__main__":
    main()
