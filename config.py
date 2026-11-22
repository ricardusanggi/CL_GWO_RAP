# path: full_code/config.py
def build_config():
    META_ALGO = "PSO"  # <-- definisikan dulu [AMGWO_GenliangLi, RLCGWO_ZhengpengHu, BFLGWO_YongWang]
    return {
        # ==================================================
        # DATASET & INPUT IMAGE
        # ==================================================
        "DATASET_DIR": "/mnt/d/Hibah BIMA/2026/BIMA_GFS/dataset/type of vegetation",
        "IMG_SIZE": (224, 224),

        # ==================================================
        # CNN BACKBONE
        # ==================================================
        "BACKBONE": "convnext_tiny", # "resnet50", "vgg16", "inceptionv3", etc.

        # ==================================================
        # METAHEURISTIC OPTIMIZATION
        # ==================================================
        "META_ALGO":  META_ALGO,
        "POP_SIZE":   25,    # digunakan jika POPULATION_LIST kosong []
        "MAX_ITER":   25,    # digunakan jika MAX_ITER_LIST kosong []

        # Variasi populasi yang dijalankan secara loop di main_loopMeta.py
        # Kosongkan [] untuk menggunakan nilai POP_SIZE di atas
        #"POPULATION_LIST": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "POPULATION_LIST": [90],
        
        # Variasi iterasi yang dijalankan secara loop di main_loopMeta.py
        # Kosongkan [] untuk menggunakan nilai MAX_ITER di atas
        #"MAX_ITER_LIST": [10],
        "MAX_ITER_LIST": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

        # ==================================================
        # CLASSIFIER CONFIGURATION
        # ==================================================
        "CLASSIFIER_MODE": "cnn",   # "ml" or "cnn"
        "CLASSIFIER": "",          # "svm", "rf", "knn", etc.
        "KNN_K": 0,                    # for KNN classifier

        # ==================================================
        # CNN HEAD PARAMETERS
        # ==================================================
        "CNN_EPOCHS": 30,
        "CNN_BATCH": 32,

        # ==================================================
        # CROSS VALIDATION
        # ==================================================
        "CV_FOLDS": 5,          # outer K-Fold
        "CNN_VAL_RATIO": 0.1,   # validation internal (CNN only)

        # ==================================================
        # SKENARIO FITNESS
        # ==================================================
        # Skenario fitur yang digunakan SAAT EVALUASI FITNESS (optimasi):
        #   "A"  →  [X_sel | f_sel]              fitur terpilih + fitur sintesis dari selected
        #   "B"  →  [X_dis | f_dis]              fitur dibuang  + fitur sintesis dari discarded
        #   "C"  →  [X_all | f_sel]              semua fitur asli + fitur sintesis dari selected
        #   "D"  →  [X_all | f_dis]              semua fitur asli + fitur sintesis dari discarded
        #   "E"  →  [X_all | f_sel | f_dis]      semua fitur asli + dua fitur sintesis  (DEFAULT)
        #   "F"  →  X_all                        semua fitur asli saja (baseline, tanpa konstruksi)
        #   "G"  →  X_sel                        hanya fitur terpilih (seleksi fitur murni)
        #   "H"  →  X_dis                        hanya fitur dibuang
        #
        # Catatan: skenario ini hanya mempengaruhi arah optimasi (fitness).
        #          Evaluasi akhir (final classification) tetap menjalankan semua skenario A–H.
        "FITNESS_SCENARIO": "G",

        # ==================================================
        # REPRODUCIBILITY
        # ==================================================
        "RANDOM_STATE": 42,

        # ==================================================
        # OUTPUT
        # ==================================================
        "RESULT_DIR": "results",
        "EXPERIMENT_NAME": f"cnn_{META_ALGO}_hybrid",
    }
