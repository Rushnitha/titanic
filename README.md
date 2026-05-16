# Titanic Survival Prediction – ML Pipeline

A clean, production-structured machine learning project that trains and evaluates three classifiers on the Titanic dataset.

## Project Structure

```
project_root/
├── data/
│   ├── raw/                   # titanic.csv  (auto-downloaded)
│   └── processed/             # titanic_clean.csv
├── plots/                     # confusion_matrices.png | roc_curves.png | feature_importances.png
├── models/                    # best_model.pkl | metrics.json
├── src/
│   ├── utils.py               # Logging, path helpers, metrics I/O
│   ├── data_prep.py           # Step 1 – Fetch & clean data
│   ├── features.py            # Step 2 – Impute / encode / scale / split
│   └── models.py              # Step 3 – Train / evaluate / save
├── tests/
│   └── test_pipeline.py       # Pytest unit tests
├── run_pipeline.py            # Master orchestrator
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python run_pipeline.py
```

This executes all five steps in sequence and prints a summary table like:

```
Model                         Accuracy       AUC
────────────────────────────────────────────────
Logistic Regression             0.8156    0.8651
Decision Tree                   0.7989    0.8327
Random Forest                   0.8380    0.8812

★ Best model (by AUC): Random Forest
```

### Run individual steps

```bash
python src/data_prep.py       # fetch + clean only
python src/features.py        # feature engineering only
python src/models.py          # full train + evaluate + save
```

## Run tests

```bash
python -m pytest tests/ -v
```

## Output artefacts

| Path | Contents |
|---|---|
| `data/raw/titanic.csv` | Original CSV from GitHub |
| `data/processed/titanic_clean.csv` | Cleaned, encoded-ready CSV |
| `models/best_model.pkl` | Best classifier (by AUC) serialised with joblib |
| `models/metrics.json` | Accuracy & AUC for every model |
| `plots/confusion_matrices.png` | Side-by-side heatmaps (3 models) |
| `plots/roc_curves.png` | Overlaid ROC curves |
| `plots/feature_importances.png` | Top-10 Random Forest features |

## Algorithms

| Model | Why |

| **Logistic Regression** | Linear baseline; fast; coefficients are directly interpretable |
| **Decision Tree** | Non-linear; human-readable rules; tendency to overfit without depth limit |
| **Random Forest** | Ensemble of trees; robust to noise; provides feature importances |

## Design decisions

- **No leakage**: `SimpleImputer` and `StandardScaler` are fit *only* on training data, then applied to test.
- **`ColumnTransformer`**: numerical and categorical branches processed in one composable object.
- **`stratify=y`**: class proportions preserved in both splits.
- **`drop='first'`** in `OneHotEncoder`: avoids the dummy-variable trap (perfect multicollinearity).