# ============================================================
# CLASSIFIER FACTORY
# - ML klasik (CPU)
# - CNN Head (PyTorch, Multi-GPU DataParallel)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ==================================================
# CLASSICAL ML CLASSIFIERS (CPU ONLY)
# ==================================================
def create_ml(cfg):
    """
    Factory untuk classifier ML klasik.
    Digunakan pada:
    - Fitness PSO
    - Final classification (mode = 'ml')

    Semua classifier di sini berjalan di CPU.
    """

    name = cfg["CLASSIFIER"].lower()

    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=cfg["KNN_K"]
        )

    if name == "svm":
        return SVC(kernel="rbf")

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.get("RF_TREES", 100),
            random_state=cfg["RANDOM_STATE"],
            n_jobs=-1
        )

    if name == "lr":
        return LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )

    raise ValueError(f"Unknown classifier: {name}")


# ==================================================
# TORCH CNN HEAD (MLP CLASSIFIER)
# - GPU ENABLED
# - MULTI-GPU via DataParallel
# ==================================================
class TorchHead(nn.Module):
    """
    CNN Head berbasis MLP (fully connected network).

    Digunakan sebagai classifier di atas:
    - CNN features
    - fitur hasil PSO (feature construction)

    Karakteristik:
    - Dilatih dari nol di setiap fold
    - Mendukung multi-GPU (DataParallel)
    - Cocok untuk dataset fitur (bukan citra mentah)
    """

    def __init__(self, input_dim, n_classes, cfg):
        super().__init__()

        # --------------------------------------------------
        # DEVICE SETUP
        # --------------------------------------------------
        # Gunakan GPU jika tersedia, jika tidak fallback ke CPU
        self.device = cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------------------------------
        # MODEL ARCHITECTURE (MLP)
        # --------------------------------------------------
        # ⚠️ JANGAN panggil .to(device) di sini
        # karena DataParallel harus membungkus model
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

        # --------------------------------------------------
        # MULTI-GPU SUPPORT (DataParallel)
        # --------------------------------------------------
        # Aktif hanya jika:
        # - CUDA tersedia
        # - GPU lebih dari 1
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)

        # --------------------------------------------------
        # PINDAHKAN MODEL KE DEVICE (SEKALI SAJA)
        # --------------------------------------------------
        self.net = self.net.to(self.device)

        # --------------------------------------------------
        # LOSS FUNCTION
        # --------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()

        # --------------------------------------------------
        # OPTIMIZER
        # --------------------------------------------------
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=cfg.get("CNN_LR", 1e-3)
        )

        # --------------------------------------------------
        # TRAINING PARAMETER
        # --------------------------------------------------
        self.epochs     = cfg["CNN_EPOCHS"]
        self.batch_size = cfg.get("CNN_BATCH", 32)

    def fit_eval(self, Xtr, ytr, Xte, yte):

        # --------------------------------------------------
        # TRAINING PHASE (mini-batch)
        # --------------------------------------------------
        Xtr_t = torch.from_numpy(Xtr).float()
        ytr_t = torch.from_numpy(ytr).long()
        n     = Xtr_t.shape[0]
        bs    = self.batch_size

        self.net.train()
        for _ in range(self.epochs):
            # shuffle indices tiap epoch
            perm = torch.randperm(n)
            for start in range(0, n, bs):
                idx = perm[start: start + bs]
                xb  = Xtr_t[idx].to(self.device, non_blocking=True)
                yb  = ytr_t[idx].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                self.optimizer.step()

        # --------------------------------------------------
        # EVALUATION PHASE
        # --------------------------------------------------
        self.net.eval()
        with torch.no_grad():
            Xte = torch.from_numpy(Xte).float().to(
                self.device, non_blocking=True
            )
            yte = torch.from_numpy(yte).long().to(
                self.device, non_blocking=True
            )
            pred = self.net(Xte).argmax(dim=1)

        # Hitung akurasi
        return (pred == yte).float().mean().item()


# ==================================================
# FACTORY FUNCTION (ENTRY POINT)
# ==================================================
def create_torch_head(input_dim, n_classes, cfg):
    """
    Factory function untuk CNN head.

    Mengembalikan:
    - TorchHead (GPU / Multi-GPU ready)
    """
    return TorchHead(input_dim, n_classes, cfg)
