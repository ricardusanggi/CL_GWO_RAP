#!/usr/bin/env python
# ============================================================
# CUDA / TORCH ENV FIX
# ============================================================
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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


# ============================================================
# MAIN
# ============================================================
def main():

    # ================= BASE CONFIG =================
    cfg_base = build_config()

    # ================= CNN FEATURE EXTRACTOR =================
    extractor = CNNFeatureExtractor(
        img_size=cfg_base["IMG_SIZE"],
        backbone=cfg_base["BACKBONE"],
        device=DEVICE
    )
    backbone, preprocess, feat_dim = extractor.build()

    # ================= LOAD DATA =================
    X_img, y = load_images(cfg_base["DATASET_DIR"], preprocess)

    print("[INFO] Extracting CNN features...")
    X_feat = backbone.predict(X_img, batch_size=512)

    # ========================================================
    # NORMALISASI LIST EKSPERIMEN (dibaca dari config.py)
    # ========================================================
    population_list = cfg_base["POPULATION_LIST"] or [cfg_base["POP_SIZE"]]
    max_iter_list   = cfg_base["MAX_ITER_LIST"]   or [cfg_base["MAX_ITER"]]

    # ========================================================
    # LOOP EKSPERIMEN
    # ========================================================
    for pop_size in population_list:
        for max_iter in max_iter_list:

            print("=" * 80)
            print(f"[EXPERIMENT] POP_SIZE={pop_size} | MAX_ITER={max_iter}")
            print("=" * 80)

            # ================= CONFIG PER EXPERIMENT =================
            cfg = cfg_base.copy()
            cfg["POP_SIZE"]        = pop_size
            cfg["MAX_ITER"]        = max_iter
            cfg["EXPERIMENT_NAME"] = (
                f"{cfg_base['EXPERIMENT_NAME']}_pop{pop_size}_iter{max_iter}"
            )

            # ================= OUTPUT DIR =================
            ensure_dir(cfg["RESULT_DIR"])
            exp_dir = os.path.join(
                cfg["RESULT_DIR"],
                f"{cfg['EXPERIMENT_NAME']}_{now()}"
            )
            ensure_dir(exp_dir)

            # ================= PSO =================
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
                "save_population": True,
            }

            optimizer = create_optimizer(cfg)

            print(f"[INFO] Running {cfg_base['META_ALGO']} optimization...")
            best = optimizer.solve(problem)

            # ================= LOGGING  =================
            logger = UnifiedLogger(exp_dir, constructor, cfg)
            logger.log_from_history(optimizer)
            logger.save_best_only()
            logger.save_population_history(optimizer)

            # ================= FINAL CLASSIFICATION =================
            decoded = constructor.decode(best.solution)
            if decoded is None:
                raise RuntimeError("Best solution tidak valid")

            f_sel, f_dis = constructor.construct_features_only(X_feat, decoded)

            final_results = run_classification(
                X_feat=X_feat,
                y=y,
                decoded=decoded,
                f_sel=f_sel,
                f_dis=f_dis,
                cfg=cfg,
                mode=cfg["CLASSIFIER_MODE"].lower()
            )

            logger.save_final_classification_results(final_results)

            print("[DONE]")
            print("BEST ACCURACY:", -best.target.fitness)
            print("RESULT DIR  :", exp_dir)

    print("=" * 80)
    print("SEMUA EKSPERIMEN SELESAI")
    print("=" * 80)


if __name__ == "__main__":
    main()
