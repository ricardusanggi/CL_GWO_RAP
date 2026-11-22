# path: full_code/core/fitness.py
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from core.classifier_factory import create_ml, create_torch_head
from utils.io import set_seed


# ==================================================
# PEMETAAN SKENARIO FITNESS
# ==================================================
# Kode skenario yang valid untuk cfg["FITNESS_SCENARIO"]
# (diset di main_loopMeta.py → bagian OPTIONAL EXPERIMENT LISTS)
#
#   "A"  →  [X_sel | f_sel]                 fitur terpilih + fitur sintesis dari selected
#   "B"  →  [X_dis | f_dis]                 fitur dibuang  + fitur sintesis dari discarded
#   "C"  →  [X_all | f_sel]                 semua fitur asli + fitur sintesis dari selected
#   "D"  →  [X_all | f_dis]                 semua fitur asli + fitur sintesis dari discarded
#   "E"  →  [X_all | f_sel | f_dis]         semua fitur asli + dua fitur sintesis  (DEFAULT)
#   "F"  →  X_all                           semua fitur asli saja (baseline)
#   "G"  →  X_sel                           hanya fitur terpilih (seleksi murni)
#   "H"  →  X_dis                           hanya fitur dibuang
#
VALID_SCENARIOS = {"A", "B", "C", "D", "E", "F", "G", "H"}


class Fitness:
    """
    Fitness function untuk optimasi metaheuristik.

    Metodologi:
    - Menggunakan Stratified K-Fold Cross-Validation (K = CV_FOLDS)
    - Nilai fitness = rata-rata akurasi seluruh fold
    - Skenario fitur dipilih melalui cfg["FITNESS_SCENARIO"] (A–H)

    ML klasik:
        * Dilatih langsung pada data training fold
        * Dievaluasi pada data test fold
    CNN:
        * Data training fold dibagi lagi menjadi:
            - data latih CNN
            - data validasi internal (CNN_VAL_RATIO)
        * Data validasi internal digunakan untuk mengurangi overfitting
    """

    def __init__(self, X, y, cfg):
        self.X        = X
        self.y        = y
        self.cfg      = cfg
        self.n_classes = len(np.unique(y))

        # Tentukan skenario sekali saat inisialisasi
        self.scenario = self._resolve_scenario()

        # Stratified K-Fold
        self.cv = StratifiedKFold(
            n_splits=self.cfg["CV_FOLDS"],
            shuffle=True,
            random_state=self.cfg["RANDOM_STATE"]
        )

        print(f"[Fitness] Skenario fitness: {self.scenario}")

    # ==================================================
    # RESOLVE SKENARIO
    # ==================================================
    def _resolve_scenario(self):
        """
        Menentukan skenario fitur yang digunakan saat evaluasi fitness.
        Skenario dibaca dari cfg["FITNESS_SCENARIO"] yang diset di main_loopMeta.py.
        Jika tidak diset atau tidak valid, default ke "E".
        """
        sc = str(self.cfg.get("FITNESS_SCENARIO", "")).strip().upper()
        return sc if sc in VALID_SCENARIOS else "E"

    # ==================================================
    # BANGUN FITUR SESUAI SKENARIO
    # ==================================================
    def _build_features(self, constructor, decoded, X):
        """
        Membangun matriks fitur sesuai skenario yang aktif.

        Parameters
        ----------
        constructor : FeatureConstructor
        decoded     : dict  —  hasil decode solusi metaheuristik
        X           : ndarray  —  data mentah untuk fold ini

        Returns
        -------
        ndarray  —  matriks fitur siap pakai
        """
        sc = self.scenario

        # ── fitur dasar ──────────────────────────────────────────────
        X_sel = X[:, decoded["selected"]]
        X_dis = X[:, decoded["discarded"]] if decoded["discarded"] else np.empty((X.shape[0], 0))

        # ── fitur sintesis ────────────────────────────────────────────
        # f_sel: chain dari fitur terpilih
        f_sel = constructor.construct_chain(
            X, decoded["selected"], decoded["operators"]
        ).reshape(-1, 1)

        # f_dis: chain dari fitur dibuang (butuh >= 2 fitur)
        if len(decoded.get("discarded", [])) >= 2:
            f_dis = constructor.construct_chain(
                X, decoded["discarded"], decoded["operators"]
            ).reshape(-1, 1)
        else:
            f_dis = np.zeros((X.shape[0], 1))

        # ── pilih skenario ────────────────────────────────────────────
        if sc == "A":
            # [X_sel | f_sel]
            return np.column_stack([X_sel, f_sel])

        elif sc == "B":
            # [X_dis | f_dis]
            return np.column_stack([X_dis, f_dis])

        elif sc == "C":
            # [X_all | f_sel]
            return np.column_stack([X, f_sel])

        elif sc == "D":
            # [X_all | f_dis]
            return np.column_stack([X, f_dis])

        elif sc == "E":
            # [X_all | f_sel | f_dis]
            return np.column_stack([X, f_sel, f_dis])

        elif sc == "F":
            # X_all saja (baseline)
            return X.copy()

        elif sc == "G":
            # X_sel saja
            return X_sel

        elif sc == "H":
            # X_dis saja
            return X_dis

        else:
            # Seharusnya tidak tercapai, fallback ke E
            return np.column_stack([X, f_sel, f_dis])

    # ==================================================
    # FITNESS = MEAN CV ACCURACY
    # ==================================================
    def evaluate(self, constructor, decoded):
        """
        Menghitung nilai fitness satu solusi metaheuristik.

        Parameters
        ----------
        constructor : FeatureConstructor
        decoded     : dict  —  solusi yang telah di-decode

        Returns
        -------
        float  —  rata-rata akurasi K-Fold
        """
        scores = []

        for tr, te in self.cv.split(self.X, self.y):

            set_seed(self.cfg["RANDOM_STATE"])

            Xtr_full, Xte = self.X[tr], self.X[te]
            ytr_full, yte = self.y[tr], self.y[te]

            # Bangun fitur sesuai skenario aktif
            Xtr_aug_full = self._build_features(constructor, decoded, Xtr_full)
            Xte_aug      = self._build_features(constructor, decoded, Xte)

            # Proteksi numerik
            if not np.isfinite(Xtr_aug_full).all():
                return 0.0

            # ── ML CLASSIFIER ────────────────────────────────────────
            if self.cfg["CLASSIFIER_MODE"] == "ml":
                clf = create_ml(self.cfg)
                clf.fit(Xtr_aug_full, ytr_full)
                acc = accuracy_score(yte, clf.predict(Xte_aug))

            # ── CNN CLASSIFIER ───────────────────────────────────────
            else:
                # Latih pada seluruh training fold (tanpa internal split)
                model = create_torch_head(
                    Xtr_aug_full.shape[1],
                    self.n_classes,
                    self.cfg
                )

                acc = model.fit_eval(
                    Xtr_aug_full, ytr_full,
                    Xte_aug, yte
                )

                del model
                torch.cuda.empty_cache()

            scores.append(acc)

        return float(np.mean(scores))
