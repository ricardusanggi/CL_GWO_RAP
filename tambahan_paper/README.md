# Eksperimen Tambahan Paper — Respon Reviewer

Folder ini berisi kode untuk 4 eksperimen tambahan sebagai respons terhadap reviewer.

## Struktur

```
tambahan_paper/
├── exp1_high_dim_backbone/
│   ├── run_high_dim.py        # Eksperimen backbone >768-dim (5 backbone)
│   └── recover_summary.py     # Recovery summary dari hasil yang sudah ada
├── exp2_non_meta_baselines/
│   └── run_baselines.py       # Baseline LASSO, mRMR, CNN-Attention
├── exp3_sensitivity/
│   ├── run_sensitivity.py     # Grid search max_flag × alpha (20 kombinasi)
│   └── recover_summary.py     # Recovery summary dari hasil yang sudah ada
├── exp4_complexity/
│   └── analyze_complexity.py  # Analisis kompleksitas teoritis + profiling
├── compile_all_results.py     # Compile semua hasil ke Excel + teks reviewer
├── build_excel_full.py        # Build Excel lengkap 7 sheet
└── reviewer_response.txt      # Narasi siap pakai menjawab reviewer (output)
```

## Cara Menjalankan

```bash
cd tambahan_paper

# Exp 1: Backbone dimensi tinggi (perlu GPU + dataset)
python exp1_high_dim_backbone/run_high_dim.py

# Exp 2: Baseline non-metaheuristik (perlu GPU + dataset)
python exp2_non_meta_baselines/run_baselines.py

# Exp 3: Sensitivitas hyperparameter (perlu GPU + dataset)
python exp3_sensitivity/run_sensitivity.py

# Exp 4: Analisis kompleksitas (tidak perlu GPU/dataset)
python exp4_complexity/analyze_complexity.py

# Compile hasil ke Excel + reviewer response
python build_excel_full.py
python compile_all_results.py
```

## Parameter

| Parameter | Nilai |
|---|---|
| Optimizer | CL_GWO |
| Backbone utama | convnext_tiny (768-dim) |
| POP_SIZE | 25 |
| MAX_ITER | 25 |
| w_min / w_max | 0.2 / 2.0 |
| max_flag (default) | 7 |
| alpha (default) | 0.3 |
| CV_FOLDS | 5 (Stratified K-Fold) |
| Classifier | SVM (rbf) |
| RANDOM_STATE | 42 |

## Hasil Utama

| Eksperimen | Temuan |
|---|---|
| Exp 1 | Sc E konsisten terburuk di semua backbone; Sc A terbaik di convnext_large |
| Exp 2 | CL_GWO kompetitif vs LASSO/mRMR/CNN-Attention; unggul di multi-objective |
| Exp 3 | Variasi akurasi hanya 1.36% di 20 kombinasi → model ROBUST |
| Exp 4 | Kompleksitas O(T·N·(D+K·C)) identik dengan GWO standar |
