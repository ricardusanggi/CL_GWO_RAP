# path: full_code/core/final_classification.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from core.classifier_factory import create_ml, create_torch_head
from utils.io import set_seed


def run_classification(
    X_feat,
    y,
    decoded,
    f_sel,
    f_dis,
    cfg,
    mode="ml"
):
    """
    Final classification evaluation using Stratified K-Fold cross-validation.

    Evaluation strategy:
    --------------------
    1. Classical Machine Learning (ML):
       - Uses Stratified K-Fold cross-validation
       - Model is trained on training folds and evaluated on test folds

    2. CNN-based classifier (TorchHead):
       - Uses Stratified K-Fold cross-validation
       - No internal validation is applied
         (TorchHead.fit_eval does not support validation yet)

    Scenarios:
    ----------
    A–E : Proposed method (feature selection + feature construction)
    F   : Baseline (original CNN features only, no feature construction)
    G   : Selected features only (no construction)
    H   : Discarded features only (no construction)

    Output:
    -------
    For each scenario:
    - mean accuracy across folds
    - standard deviation of accuracy
    - number of features used
    """

    # ==================================================
    # Dictionary to store results of each evaluation scenario
    # ==================================================
    results = {}
    n_classes = len(np.unique(y))

    skf = StratifiedKFold(
        n_splits=cfg["CV_FOLDS"],
        shuffle=True,
        random_state=cfg["RANDOM_STATE"]
    )
    # Pre-generate splits once — all scenarios share the same sample count and y
    splits = list(skf.split(X_feat, y))

    f_sel = f_sel.reshape(-1, 1)
    f_dis = f_dis.reshape(-1, 1)

    # ==================================================
    # Feature subsets based on PSO solution
    # ==================================================
    # Selected features
    X_sel = X_feat[:, decoded["selected"]]

    # Discarded features
    X_dis = X_feat[:, decoded["discarded"]]

    # ==================================================
    # Evaluation scenarios
    # ==================================================
    scenarios = {

        # --------------------------------------------------
        # A: Selected features + constructed feature (selected)
        # --------------------------------------------------
        "A_selected_plus_f_selected": np.column_stack([X_sel, f_sel]),

        # --------------------------------------------------
        # B: Discarded features + constructed feature (discarded)
        # --------------------------------------------------
        "B_discarded_plus_f_discarded": np.column_stack([X_dis, f_dis]),

        # --------------------------------------------------
        # C: All original CNN features + constructed selected feature
        # --------------------------------------------------
        "C_all_plus_f_selected": np.column_stack([X_feat, f_sel]),

        # --------------------------------------------------
        # D: All original CNN features + constructed discarded feature
        # --------------------------------------------------
        "D_all_plus_f_discarded": np.column_stack([X_feat, f_dis]),

        # --------------------------------------------------
        # E: All original CNN features + both constructed features
        # --------------------------------------------------
        "E_all_plus_f_selected_plus_f_discarded":
            np.column_stack([X_feat, f_sel, f_dis]),

        # --------------------------------------------------
        # F: BASELINE
        # All original CNN features only (no feature construction)
        # --------------------------------------------------
        "F_all_features_only": X_feat,

        # --------------------------------------------------
        # G: Selected features only (tanpa feature construction)
        # Ablasi untuk melihat kontribusi f_sel
        # --------------------------------------------------
        "G_selected_only": X_sel,

        # --------------------------------------------------
        # H: Discarded features only (tanpa feature construction)
        # Ablasi untuk melihat kontribusi f_dis
        # --------------------------------------------------
        "H_discarded_only": X_dis,
    }

    # ==================================================
    # Loop through all scenarios
    # ==================================================
    for name, X in scenarios.items():

        # Store accuracy of each fold
        accs = []

        # ==================================================
        # K-Fold loop
        # ==================================================
        for fold_i, (tr, te) in enumerate(splits):

            set_seed(cfg["RANDOM_STATE"] + fold_i)

            # Split into training and testing folds
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]

            # ===================== ML CLASSIFIER =====================
            if mode == "ml":

                # Initialize classical ML classifier
                clf = create_ml(cfg)

                # Train on training fold
                clf.fit(Xtr, ytr)

                # Evaluate on test fold
                acc = clf.score(Xte, yte)

            # ===================== CNN CLASSIFIER ====================
            else:

                # Initialize CNN head (MLP classifier)
                head = create_torch_head(
                    input_dim=X.shape[1],
                    n_classes=n_classes,
                    cfg=cfg
                )

                # Train and evaluate CNN
                # Note: No internal validation is applied
                acc = head.fit_eval(
                    Xtr, ytr,
                    Xte, yte
                )

            # Save fold accuracy
            accs.append(acc)

        # ==================================================
        # Store scenario-level results
        # ==================================================
        results[name] = {
            "mean_acc":   float(np.mean(accs)),
            "std_acc":    float(np.std(accs)),
            "n_features": X.shape[1],
            "fold_accs":  accs,          # list akurasi per fold
        }

    return results
