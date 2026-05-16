"""
models.py
---------
Step 4 – Train three classifiers, evaluate each, save the best model.

Classifiers
-----------
1. Logistic Regression  – linear baseline
2. Decision Tree        – non-linear, easily interpretable
3. Random Forest        – ensemble, highest expected accuracy

Evaluation artefacts produced
------------------------------
• Console: accuracy + classification report per model
• Console: top-5 feature importances (Random Forest)
• models/best_model.pkl    – joblib-serialised best model
• models/metrics.json      – accuracy & AUC for every model

Run standalone:
    python src/models.py
"""

import numpy as np
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc as sklearn_auc,
)

from src.utils import get_logger, ensure_dir, models_path, save_metrics

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model catalogue
# ──────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dictionary of {name: unfitted estimator}.

    Hyperparameter notes
    --------------------
    max_iter=1000   – Logistic Regression needs more iterations on real data.
    max_depth=5     – limits tree depth to avoid overfitting on small datasets.
    n_estimators=100 – 100 trees gives stable variance without excessive runtime.
    """
    return {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training + evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    feature_names: list,
):
    """
    Train every model in get_models(), print per-model metrics, and return
    a results dict plus all fitted models.

    Parameters
    ----------
    X_train / X_test  : np.ndarray
    y_train / y_test  : array-like
    feature_names     : list[str]  – column names after encoding (for importance)

    Returns
    -------
    results : dict  – {model_name: {"accuracy": float, "auc": float, ...}}
    trained : dict  – {model_name: fitted estimator}
    """
    models   = get_models()
    results  = {}
    trained  = {}

    for name, model in models.items():
        logger.info("─" * 50)
        logger.info("Training: %s", name)

        # ── Fit ──────────────────────────────────────────────────────────────
        model.fit(X_train, y_train)

        # ── Predict ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1

        # ── Core metrics ─────────────────────────────────────────────────────
        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = sklearn_auc(fpr, tpr)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "auc":      round(float(roc_auc), 4),
            "cm":       cm.tolist(),
            "fpr":      fpr.tolist(),
            "tpr":      tpr.tolist(),
        }
        trained[name] = model

        # ── Console output ───────────────────────────────────────────────────
        print(f"\n{'═' * 16}  {name}  {'═' * 16}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC AUC  : {roc_auc:.4f}")
        print()
        print(classification_report(y_test, y_pred,
                                    target_names=["Did not survive", "Survived"]))

        # ── Confusion matrix (text) ───────────────────────────────────────────
        print("  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"  {'':20s}  Pred:0   Pred:1")
        print(f"  {'Actual: 0 (no surv)':20s}  {cm[0,0]:>6d}   {cm[0,1]:>6d}")
        print(f"  {'Actual: 1 (surv)':20s}  {cm[1,0]:>6d}   {cm[1,1]:>6d}")

        # ── Feature importance (Random Forest only) ──────────────────────────
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices     = np.argsort(importances)[::-1]
            print("\n  Top-5 Feature Importances (Random Forest):")
            for rank, idx in enumerate(indices[:5], start=1):
                fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                print(f"    {rank}. {fname:<20s}  {importances[idx]:.4f}")

    return results, trained


# ──────────────────────────────────────────────────────────────────────────────
# Save best model
# ──────────────────────────────────────────────────────────────────────────────

def save_best_model(results: dict, trained: dict) -> str:
    """
    Identify the model with the highest AUC, serialise it to models/.

    Parameters
    ----------
    results : dict  – output of train_and_evaluate
    trained : dict  – output of train_and_evaluate

    Returns
    -------
    str – path to the saved .pkl file
    """
    best_name = max(results, key=lambda n: results[n]["auc"])
    best_model = trained[best_name]
    dest = models_path("best_model.pkl")
    ensure_dir(models_path())
    joblib.dump(best_model, dest)
    logger.info(
        "Best model: %s  (AUC=%.4f)  saved → %s",
        best_name, results[best_name]["auc"], dest
    )
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import here to avoid circular dependency when used as a module
    from src.data_prep  import fetch_raw_data, clean_data
    from src.features   import prepare_features

    raw = fetch_raw_data()
    df  = clean_data(raw)
    X_train, X_test, y_train, y_test, feat_names, _ = prepare_features(df)

    results, trained = train_and_evaluate(
        X_train, X_test, y_train, y_test, feat_names
    )

    save_best_model(results, trained)

    # Persist accuracy/AUC summary
    summary = {name: {"accuracy": r["accuracy"], "auc": r["auc"]}
               for name, r in results.items()}
    path = save_metrics(summary)
    logger.info("Metrics saved → %s", path)