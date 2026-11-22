# path: full_code/core/objective.py
import numpy as np


class MetaObjective:
    """
    Objective function untuk optimasi metaheuristik (PSO/GWO/dll).

    Fungsi objektif didefinisikan sebagai:
        score = accuracy - alpha * sparsity

    Keterangan:
    - accuracy  : rata-rata akurasi hasil evaluasi K-Fold
                  (skenario fitur ditentukan oleh Fitness via cfg["FITNESS_SCENARIO"])
    - sparsity  : proporsi jumlah fitur terpilih terhadap total fitur
    - alpha     : parameter penalti sparsity

    Karena library Mealpy menggunakan skema MINIMIZATION,
    maka nilai score dikembalikan dalam bentuk negatif.
    """

    def __init__(self, X, fitness, constructor, cfg, alpha=0.3):
        """
        Parameters
        ----------
        X           : numpy.ndarray  — matriks fitur asli hasil ekstraksi CNN
        fitness     : Fitness        — evaluator fitness (K-Fold accuracy)
        constructor : FeatureConstructor
        cfg         : dict           — konfigurasi eksperimen
        alpha       : float          — bobot penalti sparsity
        """
        self.X           = X
        self.fitness     = fitness
        self.constructor = constructor
        self.cfg         = cfg
        self.alpha       = alpha

    def __call__(self, sol):
        """
        Evaluasi satu solusi.

        Parameters
        ----------
        sol : numpy.ndarray  — vektor solusi (mask fitur + operator)

        Returns
        -------
        float  — nilai objektif negatif untuk minimization
        """

        # Decode solusi menjadi struktur bermakna
        decoded = self.constructor.decode(sol)

        # Solusi tidak valid → penalti besar
        if decoded is None:
            return 1e6

        # Hitung akurasi via Fitness (skenario fitur diatur di cfg["FITNESS_SCENARIO"])
        acc = self.fitness.evaluate(self.constructor, decoded)

        # Proteksi: fitness mengembalikan 0.0 jika ada NaN/Inf
        if acc == 0.0:
            return 1e6

        # Sparsity = proporsi fitur terpilih terhadap total fitur
        sparsity = len(decoded["selected"]) / self.constructor.n

        # Score (ingin dimaksimalkan)
        score = acc - self.alpha * sparsity

        # Mealpy minimization → return negatif
        return -score
