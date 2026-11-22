# ============================================================
# CNN FEATURE EXTRACTOR (RESEARCH-GRADE) - FINAL FIXED VERSION
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from torch import amp


class CNNFeatureExtractor:
    """
    CNNFeatureExtractor

    Fungsi:
    --------
    - Menggunakan CNN pretrained ImageNet sebagai feature extractor
    - Tidak melakukan fine-tuning (frozen backbone)
    - Mendukung banyak backbone (manual selection)
    - Mendukung multi-GPU
    - Mendukung mixed precision inference (AMP)

    Catatan desain:
    ---------------
    - Backbone dipilih MANUAL (satu per eksperimen)
    - Semua backbone tersedia secara eksplisit (whitelist)
    """

    def __init__(self, img_size, backbone, device):
        self.img_size = img_size                # (H, W)
        self.backbone = backbone.lower()        # konsisten
        self.device = device

    def build(self):
        """
        Build CNN feature extractor.

        Return:
        -------
        TorchFeatureModel : wrapper inference
        preprocess        : torchvision transform
        out_dim           : dimensi fitur
        """

        # ==================================================
        # BACKBONE FACTORY (AMAN & EKSPLISIT)
        # ==================================================
        BACKBONES = {
            # ================= RESNET =================
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
            "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
            "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, 2048),
            "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1, 2048),

            # ================= VGG =================
            "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, 512 * 7 * 7),
            "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1, 512 * 7 * 7),

            # ================= DENSENET =================
            "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, 1024),
            "densenet161": (models.densenet161, models.DenseNet161_Weights.IMAGENET1K_V1, 2208),
            "densenet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1, 1664),
            "densenet201": (models.densenet201, models.DenseNet201_Weights.IMAGENET1K_V1, 1920),

            # ================= EFFICIENTNET =================
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1, 1280),
            "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 1408),
            "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
            "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
            "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1, 2048),
            "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.IMAGENET1K_V1, 2304),
            "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1, 2560),

            # ================= CONVNEXT =================
            "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
            "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
            "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
            "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1, 1536),

            # ================= REGNET =================
            "regnet_y_400mf": (models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.IMAGENET1K_V1, 440),
            "regnet_y_800mf": (models.regnet_y_800mf, models.RegNet_Y_800MF_Weights.IMAGENET1K_V1, 784),
            "regnet_y_1_6gf": (models.regnet_y_1_6gf, models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1, 888),
            "regnet_x_1_6gf": (models.regnet_x_1_6gf, models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1, 912),

            # ================= MOBILENET V3 =================
            "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
            "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 960),

            # ================= INCEPTION =================
            "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1, 2048),
        }

        # --------------------------------------------------
        # Validasi backbone
        # --------------------------------------------------
        if self.backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        ctor, weights, out_dim = BACKBONES[self.backbone]

        # --------------------------------------------------
        # Instantiate pretrained model
        # (aux_logits dimatikan untuk inference)
        # --------------------------------------------------
        if self.backbone == "inception_v3":
            base = ctor(weights=weights, aux_logits=False)
        else:
            base = ctor(weights=weights)

        # --------------------------------------------------
        # Strip classifier → feature extractor
        # --------------------------------------------------
        if hasattr(base, "features"):
            model = nn.Sequential(
                base.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:
            model = nn.Sequential(
                *list(base.children())[:-1],
                nn.Flatten()
            )

        # --------------------------------------------------
        # Freeze weights (NO fine-tuning)
        # --------------------------------------------------
        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(self.device)

        # --------------------------------------------------
        # Preprocessing (Inception pakai 299x299)
        # --------------------------------------------------
        resize = 299 if self.backbone == "inception_v3" else self.img_size

        preprocess = T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return TorchFeatureModel(model, self.device), preprocess, out_dim


class TorchFeatureModel:
    """
    Wrapper inference:
    - Batch processing
    - Mixed precision (GPU only)
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, x, batch_size=512):
        """
        x : np.ndarray (N, H, W, C)
        return : np.ndarray (N, out_dim)
        """

        feats = []
        x = np.ascontiguousarray(x)

        # AMP hanya aktif jika CUDA
        use_amp = self.device.startswith("cuda")

        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(
                x[i:i + batch_size]
            ).to(self.device)

            # HWC → CHW
            batch = batch.permute(0, 3, 1, 2)

            with amp.autocast(device_type="cuda", enabled=use_amp):
                out = self.model(batch)

            feats.append(out.float().cpu().numpy())

        return np.concatenate(feats)
