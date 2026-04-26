from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor, engineer_features, load_data, split_features_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna optimization for churn models.")
    parser.add_argument("--data-path", default="telco.csv")
    parser.add_argument("--metric", choices=["roc_auc", "f1"], default="roc_auc")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def _score(metric: str, y_true, y_prob) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= 0.5).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def _build_model(model_name: str, trial: optuna.trial.Trial, random_state: int):
    if model_name == "logistic_regression":
        c = trial.suggest_float("C", 1e-3, 25.0, log=True)
        return LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, solver="lbfgs"
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 150, 900),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            random_state=random_state,
            eval_metric="logloss",
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            num_leaves=trial.suggest_int("num_leaves", 20, 200),
            max_depth=trial.suggest_int("max_depth", 3, 16),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 1200),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 3.0),
            auto_class_weights="Balanced",
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=random_state,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def optimize_model(model_name: str, X: pd.DataFrame, y: pd.Series, metric: str, trials: int, random_state: int):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial) -> float:
        fold_scores: list[float] = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = _build_model(model_name, trial, random_state)
            preprocessor = build_preprocessor(X_train)
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)

            y_prob = pipeline.predict_proba(X_valid)[:, 1]
            fold_scores.append(_score(metric, y_valid, y_prob))
        return float(sum(fold_scores) / len(fold_scores))

    study = optuna.create_study(direction="maximize", study_name=f"{model_name}_{metric}")
    study.optimize(objective, n_trials=trials)
    return study


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(args.data_path)
    processed_df = engineer_features(raw_df)
    prepared = split_features_target(processed_df)
    X, y = prepared.X, prepared.y

    model_names = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    ]

    summary_rows = []
    for model_name in model_names:
        study = optimize_model(
            model_name=model_name,
            X=X,
            y=y,
            metric=args.metric,
            trials=args.trials,
            random_state=args.random_state,
        )
        best_row = {
            "model": model_name,
            "metric": args.metric,
            "best_score": study.best_value,
            "best_trial": study.best_trial.number,
        }
        summary_rows.append(best_row)
        with open(models_dir / f"best_params_{model_name}.json", "w", encoding="utf-8") as fp:
            json.dump(study.best_params, fp, indent=2)
        joblib.dump(study, models_dir / f"optuna_study_{model_name}.joblib")

    summary_df = pd.DataFrame(summary_rows).sort_values(by="best_score", ascending=False)
    summary_df.to_csv(models_dir / "optuna_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

