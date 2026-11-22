# path: full_code_RAP/main_no_selection.py
# Transfer Learning + MLP Classifier (tanpa seleksi / konstruksi fitur)
# ============================================================
import os
import sys
import json
import csv
import time
import platform
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

from config import build_config
from utils.io import ensure_dir, now
from core.cnn_feature_extractor import CNNFeatureExtractor
from core.classifier_factory import create_torch_head


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")
if DEVICE == "cuda":
    print("[INFO] GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATA LOADER
# ============================================================
def load_images(dataset_dir, preprocess):
    X, y, labels = [], [], []
    class_names = sorted(os.listdir(dataset_dir))
    for label in class_names:
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
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return np.asarray(X), y_enc, le.classes_.tolist()


# ============================================================
# SAVE ENVIRONMENT INFO
# ============================================================
def save_environment(exp_dir):
    path = os.path.join(exp_dir, "environment.txt")
    with open(path, "w") as f:
        f.write(f"Python version : {sys.version}\n")
        f.write(f"Platform       : {platform.platform()}\n")
        f.write(f"Torch version  : {torch.__version__}\n")
        f.write(f"CUDA available : {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i}          : {torch.cuda.get_device_name(i)}\n")
                props = torch.cuda.get_device_properties(i)
                f.write(f"  Total memory : {props.total_memory / 1024**3:.2f} GB\n")
    print("[LOGGER] environment.txt disimpan")


# ============================================================
# MAIN
# ============================================================
def main():
    t_total_start = time.perf_counter()

    cfg = build_config()
    cfg["EXPERIMENT_NAME"] = f"no_sel_{cfg['BACKBONE']}_mlp"

    # Output dir
    ensure_dir(cfg["RESULT_DIR"])
    exp_dir = os.path.join(cfg["RESULT_DIR"], f"{cfg['EXPERIMENT_NAME']}_{now()}")
    ensure_dir(exp_dir)

    # Simpan config & environment
    with open(os.path.join(exp_dir, "experiment_config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    save_environment(exp_dir)

    # ================= CNN BACKBONE =================
    t0 = time.perf_counter()
    extractor = CNNFeatureExtractor(
        img_size=cfg["IMG_SIZE"],
        backbone=cfg["BACKBONE"],
        device=DEVICE
    )
    backbone, preprocess, feat_dim = extractor.build()
    t_backbone_build = time.perf_counter() - t0

    # ================= LOAD DATA =================
    t0 = time.perf_counter()
    X_img, y, class_names = load_images(cfg["DATASET_DIR"], preprocess)
    t_load = time.perf_counter() - t0
    n_samples  = X_img.shape[0]
    n_classes  = len(class_names)
    print(f"[INFO] Dataset  : {n_samples} sampel, {n_classes} kelas")
    print(f"[INFO] Kelas    : {class_names}")
    print(f"[INFO] Load time: {t_load:.2f}s")

    # Distribusi kelas
    class_dist = {class_names[i]: int((y == i).sum()) for i in range(n_classes)}

    # ================= FEATURE EXTRACTION =================
    t0 = time.perf_counter()
    print("[INFO] Extracting CNN features...")
    X_feat = backbone.predict(X_img, batch_size=512)
    t_extract = time.perf_counter() - t0
    print(f"[INFO] Feature shape  : {X_feat.shape}")
    print(f"[INFO] Extract time   : {t_extract:.2f}s")

    # ================= CROSS-VALIDATION MLP =================
    skf = StratifiedKFold(
        n_splits=cfg["CV_FOLDS"],
        shuffle=True,
        random_state=cfg["RANDOM_STATE"]
    )

    fold_records  = []   # detail per fold
    all_y_true    = []   # untuk confusion matrix global
    all_y_pred    = []

    print(f"\n[INFO] Klasifikasi MLP dengan {cfg['CV_FOLDS']}-Fold CV ...")
    t_cv_start = time.perf_counter()

    for fold_idx, (tr, te) in enumerate(skf.split(X_feat, y)):
        t_fold_start = time.perf_counter()

        Xtr, Xte = X_feat[tr], X_feat[te]
        ytr, yte = y[tr], y[te]

        head = create_torch_head(
            input_dim=feat_dim,
            n_classes=n_classes,
            cfg=cfg
        )

        # Training + eval
        t_train_start = time.perf_counter()
        acc = head.fit_eval(Xtr, ytr, Xte, yte)
        t_train = time.perf_counter() - t_train_start

        # Prediksi ulang untuk metrik detail
        head.net.eval()
        with torch.no_grad():
            Xte_t = torch.from_numpy(Xte).float().to(DEVICE, non_blocking=True)
            y_pred_fold = head.net(Xte_t).argmax(dim=1).cpu().numpy()

        t_fold = time.perf_counter() - t_fold_start

        # Metrik per fold
        prec  = precision_score(yte, y_pred_fold, average="macro", zero_division=0)
        rec   = recall_score(yte, y_pred_fold, average="macro", zero_division=0)
        f1    = f1_score(yte, y_pred_fold, average="macro", zero_division=0)
        kappa = cohen_kappa_score(yte, y_pred_fold)
        cm    = confusion_matrix(yte, y_pred_fold).tolist()

        fold_records.append({
            "fold":              fold_idx + 1,
            "n_train":           len(tr),
            "n_test":            len(te),
            "accuracy":          float(acc),
            "precision_macro":   float(prec),
            "recall_macro":      float(rec),
            "f1_macro":          float(f1),
            "kappa":             float(kappa),
            "confusion_matrix":  cm,
            "train_time_sec":    float(t_train),
            "fold_time_sec":     float(t_fold),
        })

        all_y_true.extend(yte.tolist())
        all_y_pred.extend(y_pred_fold.tolist())

        print(
            f"  Fold {fold_idx + 1:2d}/"
            f"{cfg['CV_FOLDS']}: "
            f"acc={acc:.6f}  "
            f"prec={prec:.6f}  "
            f"rec={rec:.6f}  "
            f"f1={f1:.6f}  "
            f"kappa={kappa:.4f}  "
            f"[{t_fold:.1f}s]"
        )

        del head
        torch.cuda.empty_cache()

    t_cv = time.perf_counter() - t_cv_start
    t_total = time.perf_counter() - t_total_start

    # ================= AGREGAT =================
    accs    = [r["accuracy"]         for r in fold_records]
    precs   = [r["precision_macro"]  for r in fold_records]
    recs    = [r["recall_macro"]     for r in fold_records]
    f1s     = [r["f1_macro"]         for r in fold_records]
    kappas  = [r["kappa"]            for r in fold_records]
    t_folds = [r["fold_time_sec"]    for r in fold_records]

    mean_acc   = float(np.mean(accs));   std_acc   = float(np.std(accs))
    mean_prec  = float(np.mean(precs));  std_prec  = float(np.std(precs))
    mean_rec   = float(np.mean(recs));   std_rec   = float(np.std(recs))
    mean_f1    = float(np.mean(f1s));    std_f1    = float(np.std(f1s))
    mean_kappa = float(np.mean(kappas)); std_kappa = float(np.std(kappas))

    # Confusion matrix global (semua fold digabung)
    cm_global = confusion_matrix(all_y_true, all_y_pred).tolist()
    report_global = classification_report(
        all_y_true, all_y_pred,
        target_names=class_names,
        digits=6
    )

    # ================= PRINT =================
    print("\n" + "=" * 70)
    print("[FINAL RESULTS]")
    print(f"  Mean Accuracy  : {mean_acc:.6f}  (±{std_acc:.6f})")
    print(f"  Mean Precision : {mean_prec:.6f}  (±{std_prec:.6f})")
    print(f"  Mean Recall    : {mean_rec:.6f}  (±{std_rec:.6f})")
    print(f"  Mean F1 (macro): {mean_f1:.6f}  (±{std_f1:.6f})")
    print(f"  Mean Kappa     : {mean_kappa:.6f}  (±{std_kappa:.6f})")
    print(f"  N features     : {feat_dim}")
    print(f"\n[TIMING]")
    print(f"  Load data      : {t_load:.2f}s")
    print(f"  CNN build      : {t_backbone_build:.2f}s")
    print(f"  Feature extract: {t_extract:.2f}s")
    print(f"  CV total       : {t_cv:.2f}s")
    print(f"  Per-fold avg   : {np.mean(t_folds):.2f}s")
    print(f"  TOTAL          : {t_total:.2f}s")
    print("=" * 70)
    print("\n[CLASSIFICATION REPORT - ALL FOLDS COMBINED]")
    print(report_global)

    # ================================================================
    # SIMPAN HASIL
    # ================================================================

    # 1) fold_results.csv — metrik lengkap per fold
    fold_csv = os.path.join(exp_dir, "fold_results.csv")
    with open(fold_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "n_train", "n_test",
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "kappa",
            "train_time_sec", "fold_time_sec"
        ])
        for r in fold_records:
            writer.writerow([
                r["fold"], r["n_train"], r["n_test"],
                f"{r['accuracy']:.6f}",
                f"{r['precision_macro']:.6f}",
                f"{r['recall_macro']:.6f}",
                f"{r['f1_macro']:.6f}",
                f"{r['kappa']:.6f}",
                f"{r['train_time_sec']:.4f}",
                f"{r['fold_time_sec']:.4f}",
            ])
    print(f"[LOGGER] fold_results.csv disimpan")

    # 2) summary.csv — agregat semua fold
    summary_csv = os.path.join(exp_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for metric, vals in [
            ("accuracy",         accs),
            ("precision_macro",  precs),
            ("recall_macro",     recs),
            ("f1_macro",         f1s),
            ("kappa",            kappas),
            ("fold_time_sec",    t_folds),
        ]:
            writer.writerow([
                metric,
                f"{np.mean(vals):.6f}",
                f"{np.std(vals):.6f}",
                f"{np.min(vals):.6f}",
                f"{np.max(vals):.6f}",
            ])
    print(f"[LOGGER] summary.csv disimpan")

    # 3) confusion_matrix_per_fold.json
    cm_fold_path = os.path.join(exp_dir, "confusion_matrix_per_fold.json")
    with open(cm_fold_path, "w") as f:
        json.dump(
            {f"fold_{r['fold']}": r["confusion_matrix"] for r in fold_records},
            f, indent=4
        )
    print(f"[LOGGER] confusion_matrix_per_fold.json disimpan")

    # 4) confusion_matrix_global.json + CSV
    cm_global_path = os.path.join(exp_dir, "confusion_matrix_global.json")
    with open(cm_global_path, "w") as f:
        json.dump({"class_names": class_names, "matrix": cm_global}, f, indent=4)

    cm_global_csv = os.path.join(exp_dir, "confusion_matrix_global.csv")
    with open(cm_global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for cls, row in zip(class_names, cm_global):
            writer.writerow([cls] + row)
    print(f"[LOGGER] confusion_matrix_global disimpan (json + csv)")

    # 5) classification_report.txt
    report_path = os.path.join(exp_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report (all folds combined)\n")
        f.write("=" * 60 + "\n")
        f.write(report_global)
    print(f"[LOGGER] classification_report.txt disimpan")

    # 6) timing.json — seluruh breakdown waktu
    timing = {
        "load_data_sec":       round(t_load, 4),
        "backbone_build_sec":  round(t_backbone_build, 4),
        "feature_extract_sec": round(t_extract, 4),
        "cv_total_sec":        round(t_cv, 4),
        "fold_times_sec":      [round(t, 4) for t in t_folds],
        "fold_train_times_sec": [round(r["train_time_sec"], 4) for r in fold_records],
        "total_sec":           round(t_total, 4),
        "total_hms":           str(datetime.timedelta(seconds=int(t_total))),
    }
    timing_path = os.path.join(exp_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=4)
    print(f"[LOGGER] timing.json disimpan")

    # 7) summary.json — ringkasan lengkap semua info
    summary = {
        "experiment_name":    cfg["EXPERIMENT_NAME"],
        "backbone":           cfg["BACKBONE"],
        "dataset_dir":        cfg["DATASET_DIR"],
        "n_samples":          n_samples,
        "n_classes":          n_classes,
        "class_names":        class_names,
        "class_distribution": class_dist,
        "n_features":         feat_dim,
        "cv_folds":           cfg["CV_FOLDS"],
        "epochs":             cfg["CNN_EPOCHS"],
        "random_state":       cfg["RANDOM_STATE"],
        "metrics": {
            "accuracy":        {"mean": mean_acc,   "std": std_acc},
            "precision_macro": {"mean": mean_prec,  "std": std_prec},
            "recall_macro":    {"mean": mean_rec,   "std": std_rec},
            "f1_macro":        {"mean": mean_f1,    "std": std_f1},
            "kappa":           {"mean": mean_kappa, "std": std_kappa},
        },
        "fold_accuracies":    accs,
        "timing":             timing,
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[LOGGER] summary.json disimpan")

    # ================= FINAL INFO =================
    print("\n" + "=" * 60)
    print("SELESAI")
    print(f"MEAN ACCURACY : {mean_acc:.6f} (±{std_acc:.6f})")
    print(f"TOTAL TIME    : {timing['total_hms']} ({t_total:.2f}s)")
    print(f"RESULT DIR    : {exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
