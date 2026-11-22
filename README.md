# CLGWO — Competitive Learning Grey Wolf Optimizer for Vegetation Classification

Source code for the research paper on CLGWO-based feature selection applied to vegetation image classification.

## Overview

This project implements **CLGWO (Competitive Learning Grey Wolf Optimizer)**, a metaheuristic optimization algorithm used for feature selection in CNN-based vegetation classification. The dataset consists of field photos of vegetation types (herba, perdu, pohon, semak, non-vegetasi).

## Project Structure

```
full_code_RAP/
├── config.py                          # Experiment configuration
├── main.py                            # Main entry point (single run)
├── main_auto_CLGWO_all_scenarios.py   # Auto run all CLGWO scenarios
├── main_auto_all_methods.py           # Auto run all optimization methods
├── main_ablation_CLGWO.py             # Ablation study runner
├── main_loopMeta.py                   # Loop over population/iteration variants
├── main_no_selection.py               # Baseline without feature selection
├── rerun_seed45_ablation.py           # Rerun ablation with fixed seed
├── core/
│   ├── classifier_factory.py          # ML/CNN classifier builder
│   ├── cnn_feature_extractor.py       # CNN backbone feature extraction
│   ├── feature_constructor.py         # Feature construction pipeline
│   ├── final_classification.py        # Final classifier evaluation
│   ├── fitness.py                     # Fitness function definition
│   ├── objective.py                   # Objective function wrapper
│   └── optimizer_factory.py          # Metaheuristic optimizer builder
├── utils/
│   ├── io.py                          # I/O utilities
│   └── logger.py                      # Logging utilities
└── dipakai scenario A_H/              # Scenario A–H experiment scripts
```

## CLGWO Implementation (mealpy)

The **CL-GWO (Competitive Learning Grey Wolf Optimizer)** algorithm in this research was **originally designed and implemented by the authors** (see `core/backup/_CL_GWO_RAP.py`). The algorithm has since been **contributed to and accepted into the [mealpy](https://github.com/thieu1995/mealpy) library** as `CL_GWO`.

The experiments in this repository use the mealpy version:

```python
from mealpy.swarm_based.GWO import CL_GWO
```

**Parameters used** (matching manuscript Section 3.6 / Table 4):
| Parameter | Value |
|-----------|-------|
| `w_min`   | 0.2   |
| `w_max`   | 2.0   |
| `max_flag`| 7     |

**Two key mechanisms in CL-GWO:**
- **DWL** (Dimension-Wise Learning): each dimension independently decides exploit vs. explore via learning probability `pci`
- **ALS** (Adaptive Leader Selection): when exploring, the peer leader is selected via tournament comparison (draw 2 peers, keep the better one)

**Ablation variants** (defined in `core/optimizer_factory.py`):
| Class | DWL | ALS | Description |
|-------|-----|-----|-------------|
| `CL_GWO` | ✅ | ✅ | Full model — contributed to mealpy |
| `CL_GWO_NoDWL` | ❌ | ✅ | ALS only, single decision per agent |
| `CL_GWO_NoALS` | ✅ | ❌ | DWL only, random peer selection |

> `CL_GWO_NoDWL` and `CL_GWO_NoALS` are custom ablation classes with identical hyperparameters for a fair comparison (Reviewer 2, point 5).

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- TensorFlow / Keras
- scikit-learn
- mealpy
- Pillow
- numpy

Install dependencies:
```bash
pip install torch torchvision scikit-learn mealpy Pillow numpy
```

## Configuration

Edit `config.py` to set:
- `DATASET_DIR` — path to the vegetation image dataset
- `BACKBONE` — CNN backbone (`convnext_tiny`, `resnet50`, `vgg16`, etc.)
- `META_ALGO` — optimization algorithm (e.g., `CLGWO`)
- `POP_SIZE`, `MAX_ITER` — population size and iteration count
- `CLASSIFIER_MODE` — `"cnn"` or `"ml"` (SVM, RF, KNN, etc.)

## Usage

**Single run:**
```bash
python main.py
```

**Run all CLGWO scenarios automatically:**
```bash
python main_auto_CLGWO_all_scenarios.py
```

**Ablation study:**
```bash
python main_ablation_CLGWO.py
```

**Baseline (no feature selection):**
```bash
python main_no_selection.py
```

## Scenarios

Scenarios A–H represent different CLGWO configuration variants (e.g., with/without competitive learning, different update strategies). See `dipakai scenario A_H/` for details.

## Citation

If you use this code, please cite the corresponding paper (details to be added upon publication).

---

**Authors:** Research team, Universitas Dian Nuswantoro (UDINUS)
