"""
model.py
--------
Trains and evaluates three models to predict F1 podium finishes.

Approach:
  - Binary classification: podium (1) vs no podium (0)
  - Train on 2019-2022, test on 2023-2024 (temporal split — no leakage)
  - Models: Logistic Regression, Random Forest, XGBoost
  - Handles class imbalance via class_weight / scale_pos_weight
  - Evaluates per-race podium prediction accuracy (our real-world metric)
  - Saves the best model + scaler to disk for the dashboard

Usage:
    python src/model.py

Output:
    models/best_model.pkl
    models/scaler.pkl
    models/feature_cols.pkl
    models/model_comparison.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Codespaces
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.pipeline        import Pipeline
from xgboost                 import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
PLOTS_DIR     = "data/processed/plots"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Temporal train/test split
TRAIN_SEASONS = [2019, 2020, 2021, 2022]
TEST_SEASONS  = [2023, 2024]

# Features used by the model
FEATURE_COLS = [
    "quali_position",
    "driver_champ_pos_pre",
    "driver_points_pre",
    "driver_wins_pre",
    "con_champ_pos_pre",
    "con_points_pre",
    "con_wins_pre",
    "rolling_avg_3",
    "rolling_avg_5",
    "circuit_avg_finish",
    "num_pit_stops",
    "fastest_lap_rank",
    "avg_speed_kph",
    "home_race",
    "grid_position",
]

TARGET = "podium"


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_split() -> tuple:
    print("Loading dataset...")
    df = pd.read_csv(f"{PROCESSED_DIR}/model_dataset.csv")
    print(f"  {len(df):,} rows loaded\n")

    # Fill the 36 first-race nulls in rolling features with neutral value (10)
    df["rolling_avg_3"] = df["rolling_avg_3"].fillna(10)
    df["rolling_avg_5"] = df["rolling_avg_5"].fillna(10)

    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test  = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"  Train: {len(train):,} rows ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})")
    print(f"  Test:  {len(test):,} rows  ({TEST_SEASONS[0]}–{TEST_SEASONS[-1]})\n")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET]

    return X_train, y_train, X_test, y_test, train, test, df


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return three models. All handle class imbalance natively via
    class_weight or scale_pos_weight so we don't need to oversample.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=7,   # ~85/15 class ratio
                eval_metric="logloss",
                random_state=42,
                verbosity=0
            ))
        ]),
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def podium_prediction_accuracy(model, X_test: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Our primary real-world metric: for each race, predict the top 3 drivers
    by highest podium probability, then check how many actual podium finishers
    we correctly identified.

    Score = (correctly predicted podium drivers) / (total actual podium slots)
    Perfect score = 1.0 (all 3 podium spots correctly predicted every race)
    """
    probs = model.predict_proba(X_test)[:, 1]
    test_copy = test_df.copy()
    test_copy["podium_prob"] = probs

    correct = 0
    total   = 0

    for (season, round_no), race in test_copy.groupby(["season", "round"]):
        # Top 3 by predicted probability
        predicted_podium = set(
            race.nlargest(3, "podium_prob")["driver_id"].values
        )
        # Actual podium finishers
        actual_podium = set(
            race[race["podium"] == 1]["driver_id"].values
        )
        correct += len(predicted_podium & actual_podium)
        total   += len(actual_podium)

    return correct / total if total > 0 else 0.0


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test, test_df) -> dict:
    """Train, evaluate, and return metrics for one model."""
    print(f"  Training {name}...")
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    pod_acc  = podium_prediction_accuracy(model, X_test, test_df)

    print(f"    ROC-AUC:               {roc_auc:.3f}")
    print(f"    Avg Precision (PR):    {avg_prec:.3f}")
    print(f"    Podium Pick Accuracy:  {pod_acc:.1%}\n")

    return {
        "name":         name,
        "model":        model,
        "roc_auc":      roc_auc,
        "avg_precision": avg_prec,
        "podium_accuracy": pod_acc,
        "y_pred":       y_pred,
        "y_proba":      y_proba,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(name: str, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Podium", "Podium"],
                yticklabels=["No Podium", "Podium"], ax=ax)
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/{name.lower().replace(' ', '_')}_confusion.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    Saved confusion matrix → {path}")


def plot_feature_importance(name: str, model, feature_cols: list):
    """Plot feature importance for tree-based models."""
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return

    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fi.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{name} — Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/{name.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"    Saved feature importance → {path}")


def plot_model_comparison(results: list):
    """Bar chart comparing all three models across key metrics."""
    names    = [r["name"] for r in results]
    roc      = [r["roc_auc"] for r in results]
    avg_prec = [r["avg_precision"] for r in results]
    pod_acc  = [r["podium_accuracy"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, roc,      width, label="ROC-AUC",            color="steelblue")
    ax.bar(x,         avg_prec, width, label="Avg Precision (PR)",  color="darkorange")
    ax.bar(x + width, pod_acc,  width, label="Podium Pick Accuracy", color="green")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.axhline(0.85, linestyle="--", color="gray", alpha=0.5, label="85% baseline")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  Saved model comparison → {path}")


def plot_podium_probs_sample(best_result: dict, test_df: pd.DataFrame):
    """
    For a single sample race, plot predicted podium probabilities per driver.
    Picks the last race in the test set.
    """
    probs = best_result["y_proba"]
    test_copy = test_df.copy()
    test_copy["podium_prob"] = probs

    # Pick the last race in 2024
    last_race = test_copy[test_copy["season"] == 2024].tail(1)[["season", "round"]]
    if last_race.empty:
        return
    s, r = last_race.iloc[0]["season"], last_race.iloc[0]["round"]

    race = test_copy[(test_copy["season"] == s) & (test_copy["round"] == r)].copy()
    race = race.sort_values("podium_prob", ascending=True)

    colors = ["gold" if p == 1 else "steelblue" for p in race["podium"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(race["driver_code"], race["podium_prob"], color=colors)
    ax.set_xlabel("Predicted Podium Probability")
    ax.set_title(f"Podium Probabilities — Season {int(s)} Round {int(r)}\n(gold = actual podium)")
    ax.axvline(0.5, linestyle="--", color="red", alpha=0.5)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/sample_race_probabilities.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved sample race plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("F1 Podium Predictor — Model Training")
    print("=" * 60)

    X_train, y_train, X_test, y_test, train_df, test_df, full_df = load_and_split()

    models  = get_models()
    results = []

    print("Training and evaluating models...")
    print("-" * 60)

    for name, model in models.items():
        result = evaluate_model(name, model, X_train, y_train, X_test, y_test, test_df)
        plot_confusion_matrix(name, y_test, result["y_pred"])
        plot_feature_importance(name, model, FEATURE_COLS)
        results.append(result)

    # ── Model comparison ──────────────────────────────────────────────────────
    plot_model_comparison(results)

    # ── Pick best model by podium accuracy ───────────────────────────────────
    best = max(results, key=lambda r: r["podium_accuracy"])
    print(f"\n  Best model: {best['name']} "
          f"(podium accuracy: {best['podium_accuracy']:.1%})")

    plot_podium_probs_sample(best, test_df)

    # ── Save best model, scaler, and feature list ─────────────────────────────
    print("\nSaving best model to disk...")

    with open(f"{MODELS_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    with open(f"{MODELS_DIR}/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    # Save model comparison table
    comparison = pd.DataFrame([{
        "model":            r["name"],
        "roc_auc":          round(r["roc_auc"], 4),
        "avg_precision":    round(r["avg_precision"], 4),
        "podium_accuracy":  round(r["podium_accuracy"], 4),
    } for r in results])
    comparison.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

    print(f"  Saved best_model.pkl   → {MODELS_DIR}/")
    print(f"  Saved feature_cols.pkl → {MODELS_DIR}/")
    print(f"  Saved model_comparison.csv → {MODELS_DIR}/")

    # ── Full classification report for best model ─────────────────────────────
    print(f"\nClassification Report — {best['name']}:")
    print("-" * 60)
    print(classification_report(
        y_test, best["y_pred"],
        target_names=["No Podium", "Podium"]
    ))

    print("=" * 60)
    print("Model training complete!")
    print("Next step: run dashboard.py")
    print("=" * 60)