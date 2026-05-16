"""
features.py
-----------
Step 3 – Feature engineering, imputation, encoding, scaling, and train/test split.

Everything that touches *numeric values* is learned on train only, then applied
to test – this prevents data leakage.

Run standalone:
    python src/features.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data_prep import load_clean_data, PROC_CSV
from src.utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Column lists
# ──────────────────────────────────────────────────────────────────────────────
TARGET       = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked"]


# ──────────────────────────────────────────────────────────────────────────────
# Build scikit-learn preprocessor
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Construct a ColumnTransformer that:

    Numerical columns  (Age, SibSp, Parch, Fare)
        1. Impute missing values with the *median* of the training set.
        2. Standardise to zero-mean / unit-variance.

    Categorical columns  (Pclass, Sex, Embarked)
        1. Impute missing values with the *most frequent* category.
        2. One-hot-encode; drop the first dummy to avoid multicollinearity.

    Why a ColumnTransformer?
    ------------------------
    It applies different pipelines to different column groups in a single,
    scikit-learn-compatible object that can be fit on train data only.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES),
        ],
        remainder="drop",       # silently ignore any other columns
    )
    return preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Main function: split → fit preprocessor → transform
# ──────────────────────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Take the cleaned dataframe, split into train / test, fit the preprocessor
    on the training portion, and return processed arrays.

    Parameters
    ----------
    df           : pd.DataFrame  – output of clean_data()
    test_size    : float         – fraction of data held out for testing
    random_state : int           – reproducibility seed

    Returns
    -------
    X_train  : np.ndarray  – processed training features
    X_test   : np.ndarray  – processed test features
    y_train  : pd.Series   – training labels
    y_test   : pd.Series   – test labels
    feat_names : list[str] – human-readable column names after encoding
    preprocessor : ColumnTransformer – fitted preprocessor (saved for reuse)
    """

    # ── Separate features and target ─────────────────────────────────────────
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    logger.info("Feature matrix shape before split: %s", X.shape)

    # ── Train / Test split ───────────────────────────────────────────────────
    # stratify=y  ensures both splits contain the same proportion of survivors
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Split  →  train=%d rows  |  test=%d rows  (%.0f / %.0f %%)",
        len(X_train), len(X_test),
        100 * (1 - test_size), 100 * test_size,
    )

    # ── Build & fit preprocessor (fit ONLY on train) ─────────────────────────
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)        # NO fit here!

    logger.info("Processed train shape: %s  |  test shape: %s",
                X_train_proc.shape, X_test_proc.shape)

    # ── Recover feature names for interpretability ───────────────────────────
    cat_enc   = preprocessor.named_transformers_["cat"]["onehot"]
    cat_names = cat_enc.get_feature_names_out(CAT_FEATURES).tolist()
    feat_names = NUM_FEATURES + cat_names
    logger.info("Feature names after encoding: %s", feat_names)

    return X_train_proc, X_test_proc, y_train, y_test, feat_names, preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_clean_data(PROC_CSV)
    X_train, X_test, y_train, y_test, feat_names, _ = prepare_features(df)
    logger.info("features.py complete.  feat_names=%s", feat_names)