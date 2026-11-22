#!/usr/bin/env python
# ================================================================
# RERUN SEED 45 — verifikasi ulang 4 run (4 varian x skenario G saja)
# yang memakai seed=45 pada studi ablasi Reviewer 2 (DWL vs ALS).
#
# Tujuan: mengecek apakah nilai GWO/scenario G/seed45 (0.9744) dan
# 3 run lain dengan seed yang sama bisa diulang (reproducible) atau
# berbeda jika dijalankan kembali. Skenario E TIDAK diulang (fokus
# hanya pada G sesuai permintaan). TIDAK menimpa folder hasil asli
# di results_ablation_Reviewer2_DWL_vs_ALS/ablation_<variant>_scenG_seed45_...;
# hasil rerun disimpan di sub-folder terpisah "_RERUN_seed45" di
# dalam folder yang sama, dengan EXPERIMENT_NAME berakhiran "_rerun"
# agar mudah dibedakan dari run asli.
# ================================================================

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import main_ablation_CLGWO as m

RESULT_DIR = os.path.join(m.RESULT_DIR, "_RERUN_seed45")
SEED = 45
SCENARIOS = ["G"]


def main():
    global_log_path = os.path.join(RESULT_DIR, "rerun_seed45.log")
    m.ensure_dir(RESULT_DIR)
    m.LOG_PATH = global_log_path

    device = m.init_device()
    cfg_base = m.build_config()

    m.log("[INIT] Membangun CNN backbone ...")
    extractor = m.CNNFeatureExtractor(
        img_size=cfg_base["IMG_SIZE"], backbone=cfg_base["BACKBONE"], device=device
    )
    backbone, preprocess, feat_dim = extractor.build()

    m.log("[INIT] Memuat dataset ...")
    X_img, y = m.load_images(cfg_base["DATASET_DIR"], preprocess)

    m.log("[INIT] Ekstraksi fitur CNN ...")
    X_feat = backbone.predict(X_img, batch_size=512)
    m.log(f"[INIT] X_feat: {X_feat.shape} | Kelas: {len(set(y))}\n")

    total_runs = len(m.ABLATION_VARIANTS) * len(SCENARIOS)
    m.log(f"{'='*70}\n  RERUN SEED={SEED} — verifikasi {total_runs} run ablasi DWL vs ALS (skenario G saja)\n"
          f"  Skenario: {SCENARIOS} | Varian: {list(m.ABLATION_VARIANTS.keys())}\n"
          f"  Total run: {total_runs}\n{'='*70}")

    per_run_path = os.path.join(RESULT_DIR, "rerun_seed45_summary.csv")
    header_written = os.path.exists(per_run_path)
    run_idx = 0

    for sc in SCENARIOS:
        bsk = m.SCENARIO_MAP[sc]
        for variant_name, algo in m.ABLATION_VARIANTS.items():
            run_idx += 1
            cfg = cfg_base.copy()
            cfg["FITNESS_SCENARIO"] = sc
            cfg["META_ALGO"] = algo
            cfg["POP_SIZE"] = m.ABLATION_POP_SIZE
            cfg["MAX_ITER"] = m.ABLATION_MAX_ITER
            cfg["RANDOM_STATE"] = SEED
            cfg["RESULT_DIR"] = RESULT_DIR
            cfg["EXPERIMENT_NAME"] = f"ablation_{variant_name}_scen{sc}_seed{SEED}_rerun"

            m.log(f"\n[{run_idx}/{total_runs}] RERUN scen={sc} variant={variant_name} seed={SEED}")
            try:
                acc, elapsed = m.run_one_ablation(cfg, X_feat, y, bsk)
                status = "OK"
            except Exception as e:
                import traceback
                acc, elapsed = -1.0, -1.0
                status = f"FAIL: {e}"
                m.log(f"      [FAIL] {e}")
                m.log(traceback.format_exc())

            row = {
                "fitness_scenario": sc, "variant": variant_name, "seed": SEED,
                "accuracy": acc,
                "elapsed_sec": round(elapsed, 1) if elapsed >= 0 else -1,
                "status": status,
            }
            m.append_summary_row(per_run_path, row, write_header=not header_written)
            header_written = True

    m.log(f"\n[DONE] Rerun seed=45 selesai. Hasil -> {per_run_path}")
    m.log("Bandingkan baris ini dengan baris seed=45 di ../ablation_summary.csv (data asli) secara manual.")


if __name__ == "__main__":
    main()
