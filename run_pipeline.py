"""
run_pipeline.py
---------------
Master script – runs the full ML pipeline end-to-end:

    Step 1  data_prep   → fetch + clean data
    Step 2  features    → impute / encode / scale / split
    Step 3  models      → train / evaluate / save best
    Step 4  visualize   → confusion matrices + ROC curves saved as PNGs

Usage
-----
    python run_pipeline.py

All artefacts land in:
    data/raw/           titanic.csv
    data/processed/     titanic_clean.csv
    models/             best_model.pkl  |  metrics.json
    plots/              confusion_matrices.png  |  roc_curves.png
"""

import os
import sys

# Make sure   src/   is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend – saves to file, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils      import get_logger, ensure_dir, project_root, save_metrics
from src.data_prep  import fetch_raw_data, clean_data
from src.features   import prepare_features
from src.models     import train_and_evaluate, save_best_model

logger = get_logger("pipeline")
PLOTS_DIR = os.path.join(project_root(), "plots")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict) -> str:
    """
    Plot one confusion matrix per model in a single figure row.

    Colour scale is intentionally red – bright cells = many predictions there.
    The diagonal should be brightest (correct predictions).

    Saved to: plots/confusion_matrices.png
    """
    n      = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Confusion Matrices – Titanic Survival Prediction",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = np.array(metrics["cm"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            ax=ax,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"{name}\nAccuracy: {metrics['accuracy']:.4f}", fontsize=11)
        ax.set_xlabel("Predicted Label\n(0 = Did Not Survive, 1 = Survived)")
        ax.set_ylabel("True Label\n(0 = Did Not Survive, 1 = Survived)")
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"], rotation=0)

    plt.tight_layout()
    dest = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrices saved → %s", dest)
    return dest


def plot_roc_curves(results: dict) -> str:
    """
    Overlay ROC curves for all models on one axes.

    The diagonal dashed line represents random guessing (AUC = 0.5).
    A model with a curve hugging the top-left corner is better.

    Saved to: plots/roc_curves.png
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colours = ["#e41a1c", "#377eb8", "#4daf4a"]   # red, blue, green
    for (name, metrics), colour in zip(results.items(), colours):
        ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{name}  (AUC = {metrics['auc']:.3f})",
            color=colour,
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate  (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves – Titanic Survival Prediction", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    dest = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC curves saved → %s", dest)
    return dest


def plot_feature_importances(results: dict, trained: dict,
                             feature_names: list) -> str:
    """
    Horizontal bar chart of the top-10 features from the Random Forest.

    Saved to: plots/feature_importances.png
    """
    rf_model = trained.get("Random Forest")
    if rf_model is None or not hasattr(rf_model, "feature_importances_"):
        logger.warning("No Random Forest model available – skipping importance plot.")
        return ""

    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]   # top-10
    top_names   = [feature_names[i] if i < len(feature_names)
                   else f"feat_{i}" for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_names)))[::-1]
    ax.barh(range(len(top_names)), top_vals[::-1], color=colours[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Gini Importance", fontsize=11)
    ax.set_title("Random Forest – Top-10 Feature Importances", fontsize=12,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    dest = os.path.join(PLOTS_DIR, "feature_importances.png")
    plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance plot saved → %s", dest)
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("  ML PIPELINE START – Titanic Survival Prediction")
    logger.info("=" * 60)

    ensure_dir(PLOTS_DIR)

    # ── Step 1: Data acquisition & cleaning ──────────────────────────────────
    logger.info("\n[STEP 1]  Fetch & Clean Data")
    raw_df   = fetch_raw_data()
    clean_df = clean_data(raw_df)

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    logger.info("\n[STEP 2]  Feature Engineering  (impute → encode → scale → split)")
    X_train, X_test, y_train, y_test, feat_names, preprocessor = \
        prepare_features(clean_df)

    # ── Step 3: Train & evaluate models ──────────────────────────────────────
    logger.info("\n[STEP 3]  Train & Evaluate Models")
    results, trained = train_and_evaluate(
        X_train, X_test, y_train, y_test, feat_names
    )

    # ── Step 4: Save artefacts ────────────────────────────────────────────────
    logger.info("\n[STEP 4]  Save Artefacts")
    save_best_model(results, trained)
    summary = {name: {"accuracy": r["accuracy"], "auc": r["auc"]}
               for name, r in results.items()}
    save_metrics(summary)

    # ── Step 5: Visualise ─────────────────────────────────────────────────────
    logger.info("\n[STEP 5]  Generate Plots")
    cm_path   = plot_confusion_matrices(results)
    roc_path  = plot_roc_curves(results)
    imp_path  = plot_feature_importances(results, trained, feat_names)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE – Summary")
    logger.info("=" * 60)
    logger.info("  %-28s  %8s  %8s", "Model", "Accuracy", "AUC")
    logger.info("  " + "-" * 46)
    for name, r in results.items():
        logger.info("  %-28s  %8.4f  %8.4f", name, r["accuracy"], r["auc"])
    best = max(results, key=lambda n: results[n]["auc"])
    logger.info("\n  ★ Best model (by AUC): %s", best)
    logger.info("\n  Plots saved to:  plots/")
    logger.info("  Models saved to: models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()