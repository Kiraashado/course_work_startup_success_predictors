"""Вложенная проверка моделей по набору, возрасту и характеристикам команды."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from config import config
from data_preprocessing import save_json
from utils.logger import setup_logger

OUT = config.ARTIFACTS_DIR / "modeling" / "team_2023_2026"
logger = setup_logger("modeling", log_file=config.ARTIFACTS_DIR / "modeling.log")
SEEDS = [42, 137]
OUTER_FOLDS = 5
INNER_FOLDS = 3
BATCH = ["batch_year", "batch_season"]
AGE = BATCH + ["company_age"]
TEAM = AGE + ["log_team_size", "num_founders"]
FEATURE_SETS = {
    "constant": BATCH,
    "lr_batch": BATCH,
    "lr_age": AGE,
    "lr_team": TEAM,
    "cb_age": AGE,
    "cb_team": TEAM,
    "cb_no_size": AGE + ["num_founders"],
    "cb_no_founders": AGE + ["log_team_size"],
}
MODEL_NAMES = {
    "constant": "Постоянный прогноз",
    "lr_batch": "ЛР: набор",
    "lr_age": "ЛР: набор и возраст",
    "lr_team": "ЛР: набор, возраст и команда",
    "cb_age": "CatBoost: набор и возраст",
    "cb_team": "CatBoost: набор, возраст и команда",
    "cb_no_size": "CatBoost: без размера команды",
    "cb_no_founders": "CatBoost: без числа основателей",
}
COMPARISONS = [
    ("lr_age", "lr_batch", "Возраст сверх набора, ЛР"),
    ("lr_team", "lr_age", "Команда сверх возраста и набора, ЛР"),
    ("cb_team", "cb_age", "Команда сверх возраста и набора, CatBoost"),
    ("cb_team", "cb_no_size", "Размер команды при известном числе основателей"),
    ("cb_team", "cb_no_founders", "Число основателей при известном размере команды"),
    ("cb_team", "lr_team", "CatBoost против ЛР на одинаковых признаках"),
]


def prepare_features(cohort):
    """Только детерминированные преобразования исходных признаков."""
    x = cohort[["batch_year", "batch_season", "company_age", "team_size", "num_founders"]].copy()
    x["batch_season"] = x.batch_season.fillna("unknown").astype(str)
    for col in ["batch_year", "company_age", "team_size", "num_founders"]:
        x[col] = x[col].astype(float)
    if x.team_size.dropna().lt(0).any():
        raise ValueError("Отрицательный размер команды")
    x["log_team_size"] = np.log1p(x.pop("team_size"))
    return x[TEAM]


def parameter_grid(name):
    if name == "constant":
        return [{}]
    if name.startswith("lr_"):
        return [{"C": c} for c in [0.1, 1.0, 10.0]]
    return [{"depth": depth, "iterations": n} for depth in [3, 5] for n in [200, 500]]


def make_model(name, params, seed):
    cols = FEATURE_SETS[name]
    if name == "constant":
        return DummyClassifier(strategy="prior")
    if name.startswith("cb_"):
        return CatBoostClassifier(
            **params, loss_function="Logloss", learning_rate=0.03, l2_leaf_reg=5,
            cat_features=["batch_season"], random_seed=seed, thread_count=2,
            verbose=False, allow_writing_files=False, nan_mode="Min",
        )
    curved = [c for c in ["batch_year", "company_age"] if c in cols]
    linear = [c for c in ["log_team_size", "num_founders"] if c in cols]
    numeric = curved + linear
    transforms = [
        ("age_and_year", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("spline", SplineTransformer(n_knots=4, degree=3, knots="uniform", include_bias=False)),
            ("scale", StandardScaler()),
        ]), curved),
        ("missing", MissingIndicator(features="all"), numeric),
        ("season", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["batch_season"]),
    ]
    if linear:
        transforms.append(("team", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), linear))
    return Pipeline([
        ("features", ColumnTransformer(transforms, remainder="drop")),
        ("model", LogisticRegression(C=params["C"], max_iter=3000, random_state=seed)),
    ])


def point_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def scores(y, p):
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def make_splits(y, groups, scheme, seed, n_splits):
    if scheme == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
    elif scheme == "grouped":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
    else:
        raise ValueError(f"Неизвестная схема: {scheme}")
    seen = np.zeros(len(y), dtype=int)
    for train, valid in splits:
        if set(train) & set(valid) or len(np.unique(y[valid])) != 2 or len(np.unique(y[train])) != 2:
            raise ValueError("Некорректное проверочное разбиение")
        if scheme == "grouped" and set(groups[train]) & set(groups[valid]):
            raise ValueError("Набор YC попал в обе части")
        seen[valid] += 1
    if not np.all(seen == 1):
        raise ValueError("Каждая компания должна проверяться один раз за повтор")
    return splits


def tune_and_fit(name, x, y, inner, seed):
    trials = []
    for params in parameter_grid(name):
        loss, count = 0.0, 0
        if name != "constant":
            for train, valid in inner:
                model = make_model(name, params, seed)
                model.fit(x.iloc[train], y[train])
                p = model.predict_proba(x.iloc[valid])[:, 1]
                loss += point_loss(y[valid], p).sum()
                count += len(valid)
        trials.append({"params": params, "inner_log_loss": float(loss/count) if count else None})
    best = min(trials, key=lambda row: row["inner_log_loss"] if row["inner_log_loss"] is not None else 0)
    model = make_model(name, best["params"], seed)
    model.fit(x, y)
    return model, best, trials


def paired_comparisons(predictions, n_bootstrap=2000):
    results = []
    for scheme, group in predictions.groupby("scheme"):
        wide = group.pivot(index=["repeat", "id", "y"], columns="model", values="probability")
        y = wide.index.get_level_values("y").to_numpy()
        for new, reference, label in COMPARISONS:
            difference = point_loss(y, wide[reference]) - point_loss(y, wide[new])
            by_company = pd.Series(difference, index=wide.index).groupby(level="id").mean().to_numpy()
            rng = np.random.default_rng(2026)
            boot = np.array([rng.choice(by_company, size=len(by_company), replace=True).mean()
                             for _ in range(n_bootstrap)])
            results.append({
                "scheme": scheme, "new": new, "reference": reference, "comparison": label,
                "gain_log_loss": float(by_company.mean()),
                "conditional_low": float(np.quantile(boot, .025)),
                "conditional_high": float(np.quantile(boot, .975)),
            })
    return pd.DataFrame(results)


def validate_predictions(predictions, ids):
    if predictions.duplicated(["scheme", "repeat", "model", "id"]).any():
        raise ValueError("Повторный проверочный прогноз")
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(0, 1).all():
        raise ValueError("Некорректные вероятности")
    for _, group in predictions.groupby(["scheme", "repeat", "model"]):
        if set(group.id) != set(ids):
            raise ValueError("Неполное покрытие проверочными прогнозами")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = config.PREPROCESSED_DIR / "analysis_cohort.parquet"
    cohort = pd.read_parquet(path)
    if not cohort.id.is_unique or not cohort.status_baseline.eq("active").all():
        raise ValueError("Неверная исходная когорта")
    if not cohort.snapshot_date.eq(config.BASELINE_DATE).all() or not cohort.followup_date.eq(config.FOLLOWUP_DATE).all():
        raise ValueError("Неверные даты наблюдения")
    if cohort.loc[~cohort.outcome_known, "success_binary"].notna().any():
        raise ValueError("Неизвестным исходам назначены метки")
    df = cohort.loc[cohort.outcome_known].sort_values("id").reset_index(drop=True)
    expected = df.status_followup.isin(config.SUCCESS_STATUSES).astype(int)
    if not df.success_binary.eq(expected).all():
        raise ValueError("Несогласованные целевые метки")
    x = prepare_features(df)
    y = df.success_binary.to_numpy(dtype=int)
    groups = df.batch.fillna("unknown").to_numpy(dtype=str)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dates": [config.BASELINE_DATE, config.FOLLOWUP_DATE],
        "n_cohort": len(cohort), "n_known": len(df), "positive": int(y.sum()),
        "excluded_unknown": int((~cohort.outcome_known).sum()),
        "features": FEATURE_SETS, "grids": {name: parameter_grid(name) for name in FEATURE_SETS},
        "seeds": SEEDS, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "schemes": {"stratified": "two repeats", "grouped": "one repeat, grouping by full baseline batch code"},
        "selection_metric": "pooled inner log loss, no class weighting or early stopping",
        "catboost_fixed": {"learning_rate": .03, "l2_leaf_reg": 5, "thread_count": 2},
        "logistic": "L2; cubic splines with 4 uniform training knots for age/year; median imputation and missing indicators",
        "comparisons": COMPARISONS,
        "uncertainty": "2000 paired company bootstrap replicates of fixed out-of-fold loss differences, averaged over repeats; not a confidence interval for the full training procedure",
        "shap": "cb_team, stratified repeat 0, held-out objects only; raw log odds, no feature selection",
        "scope": "retrospective internal validation after full-cohort EDA; not future-period validation",
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost"]},
        "python": platform.python_version(),
    }
    save_json(protocol, OUT / "protocol.json")
    predictions, tuning, split_rows, inner_rows, shap_rows = [], [], [], [], []
    with threadpool_limits(limits=1):
        for scheme in ["stratified", "grouped"]:
            for repeat, seed in enumerate(SEEDS if scheme == "stratified" else SEEDS[:1]):
                for fold, (train, test) in enumerate(make_splits(y, groups, scheme, seed, OUTER_FOLDS)):
                    metadata = {"scheme": scheme, "repeat": repeat, "fold": fold}
                    inner = make_splits(y[train], groups[train], scheme, seed + fold + 100, INNER_FOLDS)
                    for i in test:
                        split_rows.append({**metadata, "id": int(df.id.iloc[i]), "batch": groups[i], "y": int(y[i])})
                    for inner_fold, (_, valid) in enumerate(inner):
                        for i in train[valid]:
                            inner_rows.append({**metadata, "inner_fold": inner_fold, "id": int(df.id.iloc[i])})
                    for name, cols in FEATURE_SETS.items():
                        model, best, trials = tune_and_fit(name, x.iloc[train][cols], y[train], inner, seed + fold)
                        p = model.predict_proba(x.iloc[test][cols])[:, 1]
                        for i, probability in zip(test, p):
                            predictions.append({**metadata, "model": name, "id": int(df.id.iloc[i]),
                                                "y": int(y[i]), "probability": float(probability)})
                        for trial in trials:
                            tuning.append({**metadata, "model": name, "params": json.dumps(trial["params"], sort_keys=True),
                                           "inner_log_loss": trial["inner_log_loss"], "selected": trial is best})
                        if name == "cb_team" and scheme == "stratified" and repeat == 0:
                            pool = Pool(x.iloc[test][cols], cat_features=["batch_season"])
                            values = model.get_feature_importance(pool, type="ShapValues", thread_count=2)
                            raw = model.predict(pool, prediction_type="RawFormulaVal")
                            if not np.allclose(values.sum(axis=1), raw, atol=1e-8):
                                raise AssertionError("SHAP не воспроизводит исходный прогноз")
                            for i, row, raw_value in zip(test, values, raw):
                                shap_rows.append({"id": int(df.id.iloc[i]), "fold": fold,
                                                  **dict(zip(cols, map(float, row[:-1]))),
                                                  "expected_value": float(row[-1]), "raw_prediction": float(raw_value)})
                    logger.info("%s, повтор %d, часть %d: обучение %d, проверка %d, положительных %d",
                                scheme, repeat + 1, fold + 1, len(train), len(test), y[test].sum())
                    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv", index=False)
                    pd.DataFrame(tuning).to_csv(OUT / "tuning.csv", index=False)
                    pd.DataFrame(split_rows).to_csv(OUT / "outer_splits.csv", index=False)
                    pd.DataFrame(inner_rows).to_csv(OUT / "inner_splits.csv", index=False)
                    pd.DataFrame(shap_rows).to_csv(OUT / "shap_oof.csv", index=False)
    pred = pd.DataFrame(predictions)
    validate_predictions(pred, df.id)
    metric_rows = []
    for (scheme, repeat, name), group in pred.groupby(["scheme", "repeat", "model"]):
        metric_rows.append({"scheme": scheme, "repeat": repeat, "model": name, **scores(group.y, group.probability)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics_by_repeat.csv", index=False)
    summary = metrics.groupby(["scheme", "model"])[["log_loss", "average_precision", "roc_auc", "brier"]].mean().reset_index()
    summary.to_csv(OUT / "metrics.csv", index=False)
    paired_comparisons(pred).to_csv(OUT / "comparisons.csv", index=False)
    save_json({"completed_at": datetime.now(timezone.utc).isoformat(), "n_predictions": len(pred),
               "metrics": summary.to_dict("records")}, OUT / "results.json")
    logger.info("Завершено: %d моделей, %d проверочных прогнозов; %s", len(FEATURE_SETS), len(pred), OUT)


if __name__ == "__main__":
    main()
