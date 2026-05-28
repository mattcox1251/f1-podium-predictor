"""
model.py
--------
Improved model pipeline with:
  - Richer feature set (v2 features from feature_engineering.py)
  - TimeSeriesSplit cross-validation for robust evaluation
  - Optuna hyperparameter tuning for XGBoost and Random Forest
  - Stacked ensemble (LR + RF + XGB → Logistic meta-learner)
  - Best model saved by podium pick accuracy

Usage:
    python src/model.py

Output:
    models/best_model.pkl
    models/feature_cols.pkl
    models/model_comparison.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import TimeSeriesSplit, cross_val_score
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
PLOTS_DIR     = "data/processed/plots"
for d in [MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]   # +1 season vs v1
TEST_SEASONS  = [2024, 2025]

FEATURE_COLS = [
    "quali_position",
    "front_row",
    "driver_champ_pos_pre",
    "driver_points_pre",
    "driver_wins_pre",
    "con_champ_pos_pre",
    "con_points_pre",
    "con_wins_pre",
    "rolling_avg_3",
    "rolling_avg_5",
    "points_momentum",
    "dnf_rate_5",
    "teammate_gap_3",
    "con_pts_momentum",
    "circuit_avg_finish",
    "circuit_win_rate",
    "num_pit_stops",
    "fastest_lap_rank",
    "avg_speed_kph",
    "home_race",
    "grid_position",
]

TARGET = "podium"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_split():
    print("Loading dataset...")
    df = pd.read_csv(f"{PROCESSED_DIR}/model_dataset.csv")
    df["rolling_avg_3"]    = df["rolling_avg_3"].fillna(10)
    df["rolling_avg_5"]    = df["rolling_avg_5"].fillna(10)
    df["points_momentum"]  = df["points_momentum"].fillna(0)
    df["dnf_rate_5"]       = df["dnf_rate_5"].fillna(0)
    df["teammate_gap_3"]   = df["teammate_gap_3"].fillna(0)
    df["con_pts_momentum"] = df["con_pts_momentum"].fillna(0)
    df["circuit_win_rate"] = df["circuit_win_rate"].fillna(0)

    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test  = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"  Train: {len(train):,} rows  ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})")
    print(f"  Test:  {len(test):,} rows   ({TEST_SEASONS[0]}–{TEST_SEASONS[-1]})\n")

    return (train[FEATURE_COLS], train[TARGET],
            test[FEATURE_COLS],  test[TARGET],
            train, test, df)


# ── Podium accuracy metric ────────────────────────────────────────────────────

def podium_accuracy(model, X_test, test_df):
    probs = model.predict_proba(X_test)[:, 1]
    tmp   = test_df.copy()
    tmp["podium_prob"] = probs
    correct, total = 0, 0
    for (_, _), race in tmp.groupby(["season", "round"]):
        pred   = (race.nlargest(3, "podium_prob")
                  .sort_values("podium_prob", ascending=False)["driver_id"].tolist())
        actual = (race[race["podium"] == 1]
                  .sort_values("finish_position")["driver_id"].tolist())
        correct += sum(
            1 for pos, driver in enumerate(pred)
            if pos < len(actual) and driver == actual[pos]
        )
        total += len(actual)
    return correct / total if total > 0 else 0.0


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train, n_trials=40):
    print("  Tuning XGBoost with Optuna...")
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 600),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample       = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            scale_pos_weight= trial.suggest_float("scale_pos_weight", 3, 10),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf = Pipeline([("s", StandardScaler()), ("c", XGBClassifier(**params))])
        scores = cross_val_score(clf, X_train, y_train, cv=tscv,
                                 scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"    Best avg-precision: {study.best_value:.4f}  params: {best}")
    return Pipeline([
        ("s", StandardScaler()),
        ("c", XGBClassifier(**best, eval_metric="logloss",
                            random_state=42, verbosity=0))
    ])


def tune_random_forest(X_train, y_train, n_trials=30):
    print("  Tuning Random Forest with Optuna...")
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 100, 500),
            max_depth    = trial.suggest_int("max_depth", 4, 12),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            class_weight = "balanced", random_state=42, n_jobs=-1,
        )
        clf = Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(**params))])
        scores = cross_val_score(clf, X_train, y_train, cv=tscv,
                                 scoring="average_precision", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"    Best avg-precision: {study.best_value:.4f}  params: {best}")
    return Pipeline([
        ("s", StandardScaler()),
        ("c", RandomForestClassifier(**best, class_weight="balanced",
                                     random_state=42, n_jobs=-1))
    ])


# ── Stacking ensemble ─────────────────────────────────────────────────────────

def build_stacked_ensemble(xgb_pipe, rf_pipe, X_train, y_train):
    print("  Building stacked ensemble...")
    from sklearn.model_selection import StratifiedKFold
    lr_base = Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(class_weight="balanced",
                                 max_iter=1000, random_state=42))
    ])
    estimators = [
        ("xgb", xgb_pipe),
        ("rf",  rf_pipe),
        ("lr",  lr_base),
    ]
    # StratifiedKFold required by StackingClassifier (TimeSeriesSplit incompatible)
    # Temporal ordering is already enforced by our train/test season split
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced",
                                           max_iter=500, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=1,   # serial to avoid joblib multiprocessing issues in Codespaces
    )
    stack.fit(X_train, y_train)
    return stack


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name, model, X_train, y_train, X_test, y_test, test_df):
    if not hasattr(model, "classes_"):   # not yet fitted
        model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc     = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    pa      = podium_accuracy(model, X_test, test_df)
    print(f"  {name}")
    print(f"    ROC-AUC:              {roc:.4f}")
    print(f"    Avg Precision:        {ap:.4f}")
    print(f"    Podium Pick Accuracy: {pa:.1%}\n")
    return {"name": name, "model": model, "roc_auc": roc,
            "avg_precision": ap, "podium_accuracy": pa,
            "y_pred": y_pred, "y_proba": y_proba}


# ── Plots ─────────────────────────────────────────────────────────────────────

LAYOUT = dict(paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
              font=dict(color="#FFF", family="Inter"),
              margin=dict(l=40, r=40, t=50, b=40))

def save_feature_importance(model, feature_cols):
    try:
        clf = model.named_steps["c"] if hasattr(model, "named_steps") else None
        if clf is None or not hasattr(clf, "feature_importances_"):
            return
        fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#1A1A1A")
        fi.plot(kind="barh", ax=ax, color="#E10600")
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="white")
        ax.set_title("Feature Importance (Best Model)", color="white")
        ax.set_xlabel("Importance", color="white")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=120, facecolor="#1A1A1A")
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not save feature importance: {e}")

def save_model_comparison(results):
    names = [r["name"] for r in results]
    x     = np.arange(len(names))
    w     = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1A1A1A")
    ax.bar(x - w, [r["roc_auc"] for r in results],      w, label="ROC-AUC",       color="#E10600")
    ax.bar(x,     [r["avg_precision"] for r in results], w, label="Avg Precision", color="#FF8C00")
    ax.bar(x + w, [r["podium_accuracy"] for r in results], w, label="Podium Acc", color="#FFD700")
    ax.set_xticks(x); ax.set_xticklabels(names, color="white")
    ax.set_facecolor("#1A1A1A"); ax.tick_params(colors="white")
    ax.set_title("Model Comparison", color="white")
    ax.set_ylim(0, 1.15); ax.legend(facecolor="#2A2A2A", labelcolor="white")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=120, facecolor="#1A1A1A")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("F1 Podium Predictor — Improved Model Training")
    print("=" * 60 + "\n")

    X_train, y_train, X_test, y_test, train_df, test_df, full_df = load_and_split()

    # 1. Tune base models
    xgb_pipe = tune_xgboost(X_train, y_train, n_trials=40)
    rf_pipe  = tune_random_forest(X_train, y_train, n_trials=30)

    # 2. Fit tuned base models
    print("  Fitting tuned base models...")
    xgb_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    lr_pipe = Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)

    # 3. Build stacked ensemble
    stack = build_stacked_ensemble(xgb_pipe, rf_pipe, X_train, y_train)

    # 4. Evaluate all
    print("\nEvaluating models on test set...")
    print("-" * 60)
    results = [
        evaluate("Logistic Regression", lr_pipe,  X_train, y_train, X_test, y_test, test_df),
        evaluate("Random Forest (tuned)", rf_pipe, X_train, y_train, X_test, y_test, test_df),
        evaluate("XGBoost (tuned)",      xgb_pipe, X_train, y_train, X_test, y_test, test_df),
        evaluate("Stacked Ensemble",     stack,    X_train, y_train, X_test, y_test, test_df),
    ]

    best = max(results, key=lambda r: r["podium_accuracy"])
    print(f"Best model: {best['name']}  (podium accuracy {best['podium_accuracy']:.1%})")

    # 5. Save
    save_feature_importance(best["model"], FEATURE_COLS)
    save_model_comparison(results)

    with open(f"{MODELS_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open(f"{MODELS_DIR}/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    comparison = pd.DataFrame([{
        "model": r["name"], "roc_auc": round(r["roc_auc"], 4),
        "avg_precision": round(r["avg_precision"], 4),
        "podium_accuracy": round(r["podium_accuracy"], 4),
    } for r in results])
    comparison.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

    print(f"\nClassification Report — {best['name']}:")
    print("-" * 60)
    print(classification_report(y_test, best["y_pred"],
                                 target_names=["No Podium", "Podium"]))

    print("=" * 60)
    print("Training complete! Next: run dashboard.py")
    print("=" * 60)