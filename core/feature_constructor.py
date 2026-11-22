#path: full_code/core/feature_constructor.py
import numpy as np


class FeatureConstructor:
    """
    Feature constructor METAHEURISTIC-CORRECT

    Kelas ini bertanggung jawab untuk:
    1) Menerjemahkan solusi PSO menjadi:
       - fitur terpilih (selected)
       - fitur terbuang (discarded)
       - operator matematika untuk tiap fitur
    2) Mengkonstruksi fitur baru berbasis chain operator
    3) Menyediakan beberapa mode output:
       - untuk evaluasi ML
       - untuk logging
       - untuk analisis terpisah selected / discarded

    Karakteristik utama:
    - Jumlah fitur terpilih TIDAK dipaksa (sesuai solusi PSO)
    - 2 fitur baru = hasil akhir chain operator
    - Aman dari NaN / Inf / overflow
    """

    # Daftar operator yang bisa dipilih PSO
    OPS = ["add", "sub", "mul", "div", "max", "min", "mean", "absdiff", "identity"]

    def __init__(self, n_features):
        # Jumlah fitur asli (dimensi CNN)
        self.n = n_features

        # Jumlah operator yang tersedia
        self.n_ops = len(self.OPS)

        # Konstanta keamanan numerik
        self.EPS = 1e-6          # untuk mencegah pembagian nol
        self.MAX_VAL = 1e4       # clipping nilai ekstrim

    # ==================================================
    # Decode solusi PSO
    # ==================================================
    def decode(self, sol):
        """
        Mengubah vektor solusi PSO menjadi struktur bermakna.

        Struktur solusi:
        sol = [mask_0 ... mask_n-1 | op_0 ... op_n-1]

        mask_i > 0.5  → fitur i dipilih
        mask_i <= 0.5 → fitur i dibuang

        op_i ∈ [0, n_ops-2] → operator untuk fitur ke-i
        """

        # Validasi panjang solusi
        if len(sol) != 2 * self.n:
            return None

        # Mask fitur (selected / discarded)
        mask = sol[:self.n] > 0.5
        if mask.sum() < 2:
            # Minimal 2 fitur diperlukan untuk operasi biner
            return None

        # Decode operator dari [0,1] → indeks operator
        ops_raw = sol[self.n:]
        ops = np.clip(
            (ops_raw * (self.n_ops - 1)).astype(int),
            0,
            self.n_ops - 2          # identity disimpan sebagai fallback
        )

        # Indeks fitur terpilih dan terbuang
        selected = np.where(mask)[0].tolist()
        discarded = np.where(~mask)[0].tolist()

        # Struktur hasil decode
        return {
            "selected": selected,
            "discarded": discarded,
            "operators": ops.tolist(),

            # Operator khusus per subset
            "operators_selected": [ops[i] for i in selected],
            "operators_discarded": [ops[i] for i in discarded],

            # Nama operator (untuk interpretasi)
            "operator_names_selected": [self.OPS[ops[i]] for i in selected],
            "operator_names_discarded": [self.OPS[ops[i]] for i in discarded],
        }

    # ==================================================
    # Operator aman
    # ==================================================
    def apply_pair(self, a, b, op):
        """
        Menerapkan satu operator matematika pada dua vektor fitur.
        Dilengkapi proteksi:
        - NaN
        - Inf
        - Overflow
        """

        # Sanitasi input
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

        # Mapping operator
        if op == 0: out = a + b
        elif op == 1: out = a - b
        elif op == 2: out = a * b
        elif op == 3:
            # Pembagian aman
            denom = np.where(np.abs(b) < self.EPS, self.EPS, b)
            out = a / denom
        elif op == 4: out = np.maximum(a, b)
        elif op == 5: out = np.minimum(a, b)
        elif op == 6: out = 0.5 * (a + b)
        elif op == 7: out = np.abs(a - b)
        else:
            # identity
            out = a

        # Sanitasi output akhir
        return np.clip(
            np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0),
            -self.MAX_VAL,
            self.MAX_VAL
        )

    # ==================================================
    # Chain constructor
    # ==================================================
    def construct_chain(self, X, indices, ops):
        """
        Membangun 1 fitur baru dengan chain operator:

        f = (((X[i0] op_i0 X[i1]) op_i1 X[i2]) ...)

        Fitur terakhir menjadi output akhir.
        """

        if len(indices) < 2:
            # Tidak cukup fitur → vektor nol
            return np.zeros(X.shape[0])

        # Inisialisasi dengan fitur pertama
        cur = X[:, indices[0]]

        # Chain operator berurutan
        for i in range(len(indices) - 1):
            cur = self.apply_pair(
                cur,
                X[:, indices[i + 1]],
                ops[indices[i]]
            )

        return cur

    # ==================================================
    # 🔥 HANYA FITUR BARU
    # ==================================================
    def construct_features_only(self, X, decoded):
        """
        Digunakan untuk:
        - logging
        - analisis
        - klasifikasi akhir
        """
        f_sel = self.construct_chain(X, decoded["selected"], decoded["operators"])

        f_dis = (
            self.construct_chain(X, decoded["discarded"], decoded["operators"])
            if len(decoded["discarded"]) >= 2
            else np.zeros(X.shape[0])
        )

        return f_sel, f_dis

