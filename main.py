#path: full_code/main.py
# ============================================================
# CUDA / TORCH ENV FIX
# ============================================================
import os
import sys

# Ambil root folder project (…/full_code/..)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# Batasi GPU yang digunakan (GPU 0 dan 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Kurangi log CUDA / TensorFlow yang tidak penting
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# IMPORTS
# ============================================================
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


# ============================================================
# DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

if DEVICE == "cuda":
    print("[INFO] GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATA LOADER
# ============================================================
def load_images(dataset_dir, preprocess):
    """
    Load dataset citra dengan struktur:
    dataset/
        class_1/
        class_2/
        ...
    """
    X, y = [], []

    for label in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            try:
                img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
                img = preprocess(img)

                # CHW → HWC (agar konsisten dengan extractor)
                X.append(img.numpy().transpose(1, 2, 0))
                y.append(label)
            except Exception:
                continue

    return np.asarray(X), LabelEncoder().fit_transform(y)


# ============================================================
# MAIN
# ============================================================
def main():

    # ================= CONFIG =================
    cfg = build_config()

    # Catatan: FITNESS_SCENARIO diatur di config.py

    # ================= OUTPUT DIR =================
    ensure_dir(cfg["RESULT_DIR"])
    exp_dir = os.path.join(
        cfg["RESULT_DIR"],
        f"{cfg['EXPERIMENT_NAME']}_{now()}"
    )
    ensure_dir(exp_dir)

    # ================= CNN FEATURE EXTRACTOR =================
    extractor = CNNFeatureExtractor(
        img_size=cfg["IMG_SIZE"],
        backbone=cfg["BACKBONE"],
        device=DEVICE
    )

    backbone, preprocess, feat_dim = extractor.build()

    # ================= LOAD DATA =================
    X_img, y = load_images(cfg["DATASET_DIR"], preprocess)

    # ================= FEATURE EXTRACTION =================
    print("[INFO] Extracting CNN features...")
    X_feat = backbone.predict(X_img, batch_size=512)

    # ========================================================
    # METAHEURISTIC (PSO)
    # ========================================================
    constructor = FeatureConstructor(feat_dim)
    fitness = Fitness(X_feat, y, cfg)

    problem = {
        "obj_func": MetaObjective(
            X_feat,
            fitness,
            constructor,
            cfg,
            alpha=0.3
        ),
        "bounds": FloatVar(
            lb=[0.0] * (2 * feat_dim),
            ub=[1.0] * (2 * feat_dim)
        ),
        "minmax": "min",
        # 🔥 WAJIB DITAMBAHKAN
        "save_population": True,
    }

    optimizer = create_optimizer(cfg, callback=None)

    print("[INFO] Running PSO optimization...")
    best = optimizer.solve(problem)

    # ========================================================
    # LOGGING PSO
    # ========================================================
    logger = UnifiedLogger(exp_dir, constructor, cfg)
    logger.log_from_history(optimizer)
    logger.save_best_only()


    # 🔥 TAMBAHKAN BARIS INI
    logger.save_population_history(optimizer)
    
    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================
    decoded = constructor.decode(best.solution)
    if decoded is None:
        raise RuntimeError("Best solution tidak valid")

    f_sel, f_dis = constructor.construct_features_only(X_feat, decoded)

    mode = cfg["CLASSIFIER_MODE"].lower()
    if mode not in ["ml", "cnn"]:
        raise ValueError("CLASSIFIER_MODE harus 'ml' atau 'cnn'")

    print(f"\n[FINAL CLASSIFICATION MODE: {mode.upper()}]")

    final_results = run_classification(
        X_feat=X_feat,
        y=y,
        decoded=decoded,
        f_sel=f_sel,
        f_dis=f_dis,
        cfg=cfg,
        mode=mode
    )

    # ================= PRINT RESULTS =================
    print("\n[FINAL CLASSIFICATION RESULTS]")
    for name, res in final_results.items():

        if "accuracy" in res:
            acc_str = f"{res['accuracy']:.6f}"
        elif "mean_accuracy" in res:
            acc_str = (
                f"{res['mean_accuracy']:.6f} "
                f"(±{res.get('std_accuracy', 0.0):.6f})"
            )
        else:
            acc_str = "N/A"

        print(
            f"{name}: "
            f"accuracy={acc_str}, "
            f"n_features={res.get('n_features', 'N/A')}"
        )

    # ================= SAVE RESULTS =================
    logger.save_final_classification_results(final_results)


    # ================= FINAL INFO =================
    print("=" * 60)
    print("SELESAI")
    print("BEST ACCURACY (PSO):", -best.target.fitness)
    print("RESULT DIR         :", exp_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
