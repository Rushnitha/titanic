"""
tests/test_pipeline.py
----------------------
Unit tests covering each module independently.

Run with:
    python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.utils    import ensure_dir, save_metrics, load_metrics
from src.data_prep import clean_data, PROC_CSV
from src.features  import build_preprocessor, prepare_features, TARGET
from src.models    import get_models, train_and_evaluate


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Minimal synthetic dataframe that mimics the Titanic schema."""
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "Survived": np.random.randint(0, 2, n),
        "Pclass":   np.random.choice(["1", "2", "3"], n),
        "Sex":      np.random.choice([0, 1], n),
        "Age":      np.where(np.random.rand(n) < 0.2, np.nan,
                             np.random.uniform(1, 80, n)),
        "SibSp":    np.random.randint(0, 5, n),
        "Parch":    np.random.randint(0, 4, n),
        "Fare":     np.random.uniform(5, 500, n),
        "Embarked": np.random.choice(["C", "Q", "S", None], n),
    })


@pytest.fixture
def prepared(raw_df):
    """Return fully prepared train/test arrays from the synthetic dataframe."""
    return prepare_features(raw_df)


# ──────────────────────────────────────────────────────────────────────────────
# utils
# ──────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_ensure_dir_creates(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "c")
        ensure_dir(target)
        assert os.path.isdir(target)

    def test_save_and_load_metrics(self, tmp_path, monkeypatch):
        # Redirect models_path to tmp_path
        import src.utils as utils_mod
        monkeypatch.setattr(utils_mod, "models_path",
                            lambda *p: str(tmp_path.joinpath(*p)))
        path = save_metrics({"acc": 0.85}, "test_metrics.json")
        loaded = load_metrics("test_metrics.json")
        assert loaded["acc"] == pytest.approx(0.85)


# ──────────────────────────────────────────────────────────────────────────────
# data_prep
# ──────────────────────────────────────────────────────────────────────────────

class TestDataPrep:
    def test_clean_removes_target_not_dropped(self, raw_df):
        """Survived column must survive clean_data (it is not a drop column)."""
        result = clean_data(raw_df, dest=os.devnull)
        assert TARGET in result.columns

    def test_clean_shape(self, raw_df):
        result = clean_data(raw_df, dest=os.devnull)
        # Should have exactly the columns in the cleaned schema
        expected_cols = {"Survived", "Pclass", "Sex", "Age",
                         "SibSp", "Parch", "Fare", "Embarked"}
        assert expected_cols == set(result.columns)


# ──────────────────────────────────────────────────────────────────────────────
# features
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_split_sizes(self, prepared):
        X_train, X_test, y_train, y_test, *_ = prepared
        total = len(y_train) + len(y_test)
        assert abs(len(y_test) / total - 0.20) < 0.03   # roughly 20 %

    def test_no_nan_in_output(self, prepared):
        X_train, X_test, *_ = prepared
        assert not np.isnan(X_train).any(), "NaN found in X_train after preprocessing"
        assert not np.isnan(X_test).any(),  "NaN found in X_test after preprocessing"

    def test_feature_count_matches_names(self, prepared):
        X_train, _, _, _, feat_names, _ = prepared
        assert X_train.shape[1] == len(feat_names)

    def test_preprocessor_not_refit_on_test(self, raw_df):
        """Transforming twice must yield same result (idempotent test proxy)."""
        X_tr, X_te1, *_ = prepare_features(raw_df)
        X_tr2, X_te2, *_ = prepare_features(raw_df)
        np.testing.assert_array_almost_equal(X_te1, X_te2)


# ──────────────────────────────────────────────────────────────────────────────
# models
# ──────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_all_models_present(self):
        models = get_models()
        assert set(models.keys()) == {
            "Logistic Regression", "Decision Tree", "Random Forest"
        }

    def test_results_keys(self, prepared):
        X_tr, X_te, y_tr, y_te, feat_names, _ = prepared
        results, trained = train_and_evaluate(X_tr, X_te, y_tr, y_te, feat_names)
        for name in get_models():
            assert name in results
            assert "accuracy" in results[name]
            assert "auc"      in results[name]

    def test_accuracy_in_range(self, prepared):
        X_tr, X_te, y_tr, y_te, feat_names, _ = prepared
        results, _ = train_and_evaluate(X_tr, X_te, y_tr, y_te, feat_names)
        for name, r in results.items():
            assert 0.0 <= r["accuracy"] <= 1.0, f"{name} accuracy out of range"
            assert 0.0 <= r["auc"]      <= 1.0, f"{name} AUC out of range"

    def test_confusion_matrix_shape(self, prepared):
        X_tr, X_te, y_tr, y_te, feat_names, _ = prepared
        results, _ = train_and_evaluate(X_tr, X_te, y_tr, y_te, feat_names)
        for name, r in results.items():
            cm = np.array(r["cm"])
            assert cm.shape == (2, 2), f"{name} CM wrong shape"