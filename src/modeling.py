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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from config import config
from data_preprocessing import save_json
from utils.logger import setup_logger

OUT = config.ARTIFACTS_DIR / "modeling" / "team_2023_2026"
logger = setup_logger("modeling", log_file=config.ARTIFACTS_DIR / "modeling.log")
SCHEME_SEEDS = {
    "stratified": [42, 137, 271, 509, 887],
    "grouped": [42, 137, 271, 509, 887],
}
OUTER_FOLDS = 5
INNER_FOLDS = 3
BATCH = ["batch_year", "batch_season"]
AGE = BATCH + ["company_age"]
TEAM = AGE + ["log_team_size", "num_founders"]
GEO = TEAM + ["country"]
TAGS = TEAM + ["tag_text"]
TEXT = TEAM + ["short_description"]
ALL = TEAM + ["country", "tag_text", "short_description"]
FEATURE_SETS = {
    "constant": BATCH,
    "lr_base": AGE,
    "lr_team": TEAM,
    "lr_geo": GEO,
    "lr_tags": TAGS,
    "lr_text": TEXT,
    "lr_all": ALL,
    "lr_no_tags": [c for c in ALL if c != "tag_text"],
    "lr_no_text": [c for c in ALL if c != "short_description"],
    "lr_no_geo": [c for c in ALL if c != "country"],
    "lr_no_team": [c for c in ALL if c not in {"log_team_size", "num_founders"}],
    "lr_all_ap": ALL,
    "lr_all_balanced": ALL,
    "cb_team": TEAM,
}
MODEL_NAMES = {
    "constant": "Постоянный прогноз",
    "lr_base": "ЛР: набор и возраст",
    "lr_team": "ЛР: набор, возраст и команда",
    "lr_geo": "ЛР: команда и география",
    "lr_tags": "ЛР: команда и теги",
    "lr_text": "ЛР: команда и краткое описание",
    "lr_all": "ЛР: все блоки признаков",
    "lr_no_tags": "ЛР: все блоки без тегов",
    "lr_no_text": "ЛР: все блоки без описания",
    "lr_no_geo": "ЛР: все блоки без страны",
    "lr_no_team": "ЛР: все блоки без команды",
    "lr_all_ap": "ЛР: все блоки, подбор по AP",
    "lr_all_balanced": "ЛР: все блоки, веса классов",
    "cb_team": "CatBoost: набор, возраст и команда",
}
COMPARISONS = [
    ("lr_team", "lr_base", "Характеристики команды"),
    ("lr_geo", "lr_team", "География"),
    ("lr_tags", "lr_team", "Тематические теги"),
    ("lr_text", "lr_team", "Краткое описание"),
    ("lr_all", "lr_team", "География, теги и описание совместно"),
    ("cb_team", "lr_team", "CatBoost против ЛР на одинаковых признаках"),
    ("lr_all", "constant", "Полная модель против постоянного прогноза"),
    ("lr_all", "lr_tags", "Полная модель против модели с тегами"),
    ("lr_all", "lr_no_tags", "Исключение тегов из полной модели"),
    ("lr_all", "lr_no_text", "Исключение описания из полной модели"),
    ("lr_all", "lr_no_geo", "Исключение страны из полной модели"),
    ("lr_all", "lr_no_team", "Исключение команды из полной модели"),
]

RANKING_MODELS = {"lr_all_ap", "lr_all_balanced"}
SENSITIVITY_MODELS = ["constant", "lr_base", "lr_team", "lr_tags", "lr_all", "cb_team"]


def split_tags(text):
    """Сохраняет целую метку, включая пробелы и знаки пунктуации."""
    return text.split("\x1f") if text else []


def prepare_features(cohort):
    """Только детерминированные преобразования исходных признаков."""
    source = ["batch_year", "batch_season", "company_age", "team_size", "num_founders",
              "country", "tags", "short_description"]
    x = cohort.reindex(columns=source).copy()
    x["batch_season"] = x.batch_season.fillna("unknown").astype(str)
    x["country"] = x.country.fillna("unknown").astype(str)
    x["short_description"] = x.short_description.fillna("").astype(str)
    x["tag_text"] = x.tags.map(
        lambda values: "\x1f".join(sorted({str(value).strip() for value in values}))
        if isinstance(values, (list, tuple, np.ndarray)) else ""
    )
    for col in ["batch_year", "company_age", "team_size", "num_founders"]:
        x[col] = x[col].astype(float)
    if x.team_size.dropna().lt(0).any():
        raise ValueError("Отрицательный размер команды")
    x["log_team_size"] = np.log1p(x.pop("team_size"))
    return x[ALL]


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
        ("season", OneHotEncoder(handle_unknown="ignore"), ["batch_season"]),
    ]
    if linear:
        transforms.append(("team", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), linear))
    if "country" in cols:
        transforms.append(("country", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), ["country"]))
    if "tag_text" in cols:
        transforms.append(("tags", TfidfVectorizer(binary=True, use_idf=False, norm=None, min_df=5,
                                                     tokenizer=split_tags, token_pattern=None, lowercase=False), "tag_text"))
    if "short_description" in cols:
        transforms.append(("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                                            min_df=5, max_features=5000, sublinear_tf=True),
                           "short_description"))
    return Pipeline([
        ("features", ColumnTransformer(transforms, remainder="drop")),
        ("model", LogisticRegression(C=params["C"], max_iter=3000, random_state=seed,
                                     solver="liblinear",
                                     class_weight="balanced" if name.endswith("_balanced") else None)),
    ])


def point_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def scores(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    k = max(1, int(np.ceil(len(y) * .1)))
    threshold = np.partition(p, -k)[-k]
    above, tied = p > threshold, p == threshold
    remaining = k - int(above.sum())
    expected_tp = float(y[above].sum())
    if remaining and tied.sum():
        expected_tp += float(y[tied].sum()) * remaining / int(tied.sum())
    precision_at_10 = expected_tp / k
    recall_at_10 = expected_tp / y.sum()
    prevalence = float(y.mean())
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "selected_n": k,
        "selected_positive": expected_tp,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "prevalence": prevalence,
        "precision_at_10": precision_at_10,
        "recall_at_10": recall_at_10,
        "lift_at_10": precision_at_10 / prevalence,
    }


def make_splits(y, groups, scheme, seed, n_splits):
    if scheme == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
    elif scheme == "grouped":
        # Randomize entire groups, balancing both sample counts and events.
        # Candidate choice uses only fold composition, never model performance.
        unique, inverse = np.unique(groups, return_inverse=True)
        if len(unique) < n_splits:
            raise ValueError("Недостаточно наборов для групповой проверки")
        counts = np.column_stack([np.bincount(inverse), np.bincount(inverse, weights=y)])
        target = counts.sum(axis=0) / n_splits
        rng = np.random.default_rng(seed)
        best, best_cost = None, np.inf
        for _ in range(128):
            totals = np.zeros((n_splits, 2))
            assignment = np.empty(len(unique), dtype=int)
            for g in rng.permutation(len(unique)):
                delta = (((totals + counts[g]) / target) ** 2 - (totals / target) ** 2).sum(axis=1)
                choices = np.flatnonzero(np.isclose(delta, delta.min()))
                f = int(rng.choice(choices))
                assignment[g] = f
                totals[f] += counts[g]
            if np.any(totals[:, 1] == 0) or np.any(totals[:, 0] == totals[:, 1]):
                continue
            cost = np.square(totals / target - 1).sum()
            if cost < best_cost:
                best, best_cost = assignment.copy(), cost
        if best is None:
            raise ValueError("Не удалось построить части с обоими классами")
        labels = best[inverse]
        splits = [(np.flatnonzero(labels != f), np.flatnonzero(labels == f)) for f in range(n_splits)]
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
        inner_scores, sizes = [], []
        if name != "constant":
            for train, valid in inner:
                model = make_model(name, params, seed)
                model.fit(x.iloc[train], y[train])
                p = model.predict_proba(x.iloc[valid])[:, 1]
                inner_scores.append([log_loss(y[valid], p, labels=[0, 1]), average_precision_score(y[valid], p)])
                sizes.append(len(valid))
        if inner_scores:
            inner_log_loss, inner_ap = map(float, np.average(inner_scores, axis=0, weights=sizes))
        else:
            inner_log_loss, inner_ap = None, None
        trials.append({"params": params, "inner_log_loss": inner_log_loss,
                       "inner_average_precision": inner_ap})
    if name == "constant":
        best = trials[0]
    elif name in RANKING_MODELS:
        best = max(trials, key=lambda row: row["inner_average_precision"])
    else:
        best = min(trials, key=lambda row: row["inner_log_loss"])
    model = make_model(name, best["params"], seed)
    model.fit(x, y)
    return model, best, trials


def paired_comparisons(predictions, n_bootstrap=2000):
    results = []
    for scheme, group in predictions.groupby("scheme"):
        wide = group.pivot(index=["repeat", "id", "y"], columns="model", values="probability")
        y = wide.index.get_level_values("y").to_numpy()
        for new, reference, label in COMPARISONS:
            if new not in wide or reference not in wide:
                continue
            difference = point_loss(y, wide[reference]) - point_loss(y, wide[new])
            losses = pd.Series(difference, index=wide.index)
            by_id = losses.groupby(level="id").mean()
            by_company = by_id.to_numpy()
            rng = np.random.default_rng(2026)
            boot = np.array([rng.choice(by_company, size=len(by_company), replace=True).mean()
                             for _ in range(n_bootstrap)])
            batches = group.drop_duplicates("id").set_index("id").batch
            cluster = by_id.to_frame("difference").join(batches).groupby("batch").difference.agg(["sum", "size"])
            draws = rng.integers(0, len(cluster), size=(n_bootstrap, len(cluster)))
            cluster_boot = cluster["sum"].to_numpy()[draws].sum(axis=1) / cluster["size"].to_numpy()[draws].sum(axis=1)
            repeats = losses.groupby(level="repeat").mean()
            results.append({
                "scheme": scheme, "new": new, "reference": reference, "comparison": label,
                "gain_log_loss": float(by_company.mean()),
                "conditional_low": float(np.quantile(boot, .025)),
                "conditional_high": float(np.quantile(boot, .975)),
                "cluster_low": float(np.quantile(cluster_boot, .025)),
                "cluster_high": float(np.quantile(cluster_boot, .975)),
                "repeat_min": float(repeats.min()), "repeat_max": float(repeats.max()),
                "positive_repeats": int(repeats.gt(0).sum()), "n_repeats": len(repeats),
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


def aggregate_scores(group):
    """Average proper/ranking scores by fold size; aggregate selection counts."""
    result = {c: float(np.average(group[c], weights=group.n))
              for c in ["log_loss", "average_precision", "roc_auc", "brier"]}
    n, positive = int(group.n.sum()), int(group.positive.sum())
    selected, tp = int(group.selected_n.sum()), float(group.selected_positive.sum())
    result.update(n=n, positive=positive, selected_n=selected, selected_positive=tp,
                  prevalence=positive / n, precision_at_10=tp / selected,
                  recall_at_10=tp / positive, lift_at_10=(tp / selected) / (positive / n))
    return result


def split_diagnostics(rows):
    results = []
    for scheme, data in rows.groupby("scheme"):
        wide = data.pivot(index="id", columns="repeat", values="fold")
        for a in wide.columns:
            for b in wide.columns:
                if a < b:
                    results.append({"scheme": scheme, "repeat_a": a, "repeat_b": b,
                                    "adjusted_rand": adjusted_rand_score(wide[a], wide[b])})
    return pd.DataFrame(results)


def run_experiment(df, out, models, schemes, interpret=False):
    out.mkdir(parents=True, exist_ok=True)
    x = prepare_features(df)
    y = df.success_binary.to_numpy(dtype=int)
    groups = df.batch.fillna("unknown").to_numpy(dtype=str)
    predictions, tuning, split_rows, inner_rows, shap_rows, coefficient_rows = [], [], [], [], [], []
    for scheme, seeds in schemes.items():
        for repeat, seed in enumerate(seeds):
            for fold, (train, test) in enumerate(make_splits(y, groups, scheme, seed, OUTER_FOLDS)):
                metadata = {"scheme": scheme, "repeat": repeat, "fold": fold}
                inner = make_splits(y[train], groups[train], scheme, seed + fold + 100, INNER_FOLDS)
                for i in test:
                    split_rows.append({**metadata, "id": int(df.id.iloc[i]), "batch": groups[i], "y": int(y[i])})
                for inner_fold, (_, valid) in enumerate(inner):
                    for i in train[valid]:
                        inner_rows.append({**metadata, "inner_fold": inner_fold, "id": int(df.id.iloc[i])})
                for name in models:
                    cols = FEATURE_SETS[name]
                    model, best, trials = tune_and_fit(name, x.iloc[train][cols], y[train], inner, seed + fold)
                    p = model.predict_proba(x.iloc[test][cols])[:, 1]
                    for i, probability in zip(test, p):
                        predictions.append({**metadata, "model": name, "id": int(df.id.iloc[i]), "batch": groups[i],
                                            "y": int(y[i]), "probability": float(probability)})
                    for trial in trials:
                        tuning.append({**metadata, "model": name, "params": json.dumps(trial["params"], sort_keys=True),
                                       "inner_log_loss": trial["inner_log_loss"],
                                       "inner_average_precision": trial["inner_average_precision"],
                                       "selection_metric": "average_precision" if name in RANKING_MODELS else "log_loss",
                                       "selected": trial is best})
                    if interpret and name == "lr_tags" and scheme == "grouped":
                        transform = model.named_steps["features"]
                        tags = transform.named_transformers_["tags"].get_feature_names_out()
                        coef = model.named_steps["model"].coef_[0][transform.output_indices_["tags"]]
                        coefficient_rows.extend({**metadata, "tag": tag, "coefficient": float(value)}
                                                for tag, value in zip(tags, coef))
                    if interpret and name == "cb_team" and scheme == "grouped" and repeat == 0:
                        pool = Pool(x.iloc[test][cols], cat_features=["batch_season"])
                        values = model.get_feature_importance(pool, type="ShapValues", thread_count=2)
                        raw = model.predict(pool, prediction_type="RawFormulaVal")
                        if not np.allclose(values.sum(axis=1), raw, atol=1e-8):
                            raise AssertionError("SHAP не воспроизводит исходный прогноз")
                        for i, row, raw_value in zip(test, values, raw):
                            shap_rows.append({"id": int(df.id.iloc[i]), "fold": fold,
                                              **dict(zip(cols, map(float, row[:-1]))),
                                              "expected_value": float(row[-1]), "raw_prediction": float(raw_value)})
                logger.info("%s: %s, повтор %d, часть %d: обучение %d, проверка %d, положительных %d",
                            out.name, scheme, repeat + 1, fold + 1, len(train), len(test), y[test].sum())
            # Checkpoint after each complete repeat; results.json is written only on completion.
            pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
            pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)
            pd.DataFrame(split_rows).to_csv(out / "outer_splits.csv", index=False)
            pd.DataFrame(inner_rows).to_csv(out / "inner_splits.csv", index=False)
    pred = pd.DataFrame(predictions)
    validate_predictions(pred, df.id)
    diagnostics = split_diagnostics(pd.DataFrame(split_rows))
    diagnostics.to_csv(out / "split_diagnostics.csv", index=False)
    if diagnostics.loc[diagnostics.scheme.eq("grouped"), "adjusted_rand"].ge(.8).any():
        raise AssertionError("Групповые повторы недостаточно различаются")
    fold_rows = [{"scheme": s, "repeat": r, "fold": f, "model": m, **scores(g.y, g.probability)}
                 for (s, r, f, m), g in pred.groupby(["scheme", "repeat", "fold", "model"])]
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(out / "metrics_by_fold.csv", index=False)
    repeat_rows = [{"scheme": s, "repeat": r, "model": m, **aggregate_scores(g)}
                   for (s, r, m), g in fold_metrics.groupby(["scheme", "repeat", "model"])]
    metrics = pd.DataFrame(repeat_rows)
    metrics.to_csv(out / "metrics_by_repeat.csv", index=False)
    columns = [c for c in metrics if c not in {"scheme", "repeat", "model"}]
    summary = metrics.groupby(["scheme", "model"])[columns].mean().reset_index()
    summary.to_csv(out / "metrics.csv", index=False)
    paired_comparisons(pred).to_csv(out / "comparisons.csv", index=False)
    # Cohort-level diagnostics preserve the company-weighted estimand.
    wide = pred.pivot(index=["scheme", "repeat", "id", "batch", "y"], columns="model", values="probability")
    by_batch = []
    yy = wide.index.get_level_values("y").to_numpy()
    for new, reference, label in COMPARISONS:
        if new in wide and reference in wide:
            delta = pd.Series(point_loss(yy, wide[reference]) - point_loss(yy, wide[new]), index=wide.index)
            for (scheme, batch), values in delta.groupby(level=["scheme", "batch"]):
                by_batch.append({"scheme": scheme, "batch": batch, "comparison": label,
                                 "gain_log_loss": values.mean(), "n": values.index.get_level_values("id").nunique()})
    pd.DataFrame(by_batch).to_csv(out / "gains_by_batch.csv", index=False)
    if interpret:
        pd.DataFrame(shap_rows).to_csv(out / "shap_oof.csv", index=False)
        coefficients = pd.DataFrame(coefficient_rows)
        coefficients.to_csv(out / "tag_coefficients.csv", index=False)
        rows = []
        for tag, values in coefficients.groupby("tag"):
            mask = x.tag_text.map(lambda text: tag in split_tags(text))
            v = values.coefficient
            rows.append({"tag": tag, "n": int(mask.sum()), "positive": int(y[mask].sum()),
                         "fits_present": len(v), "median": v.median(), "q10": v.quantile(.1),
                         "q90": v.quantile(.9), "positive_fraction": v.gt(0).mean()})
        pd.DataFrame(rows).sort_values(["n", "tag"], ascending=[False, True]).to_csv(out / "tag_summary.csv", index=False)
    save_json({"completed_at": datetime.now(timezone.utc).isoformat(), "n_predictions": len(pred),
               "n_known": len(df), "positive": int(y.sum()), "metrics": summary.to_dict("records")}, out / "results.json")
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = config.PREPROCESSED_DIR / "analysis_cohort.parquet"
    cohort = pd.read_parquet(path)
    if not cohort.id.is_unique or not cohort.status_baseline.eq("active").all():
        raise ValueError("Неверная исходная когорта")
    if not cohort.snapshot_date.eq(config.BASELINE_DATE).all() or not cohort.followup_date.eq(config.FOLLOWUP_DATE).all():
        raise ValueError("Неверные обозначения срезов")
    if cohort.loc[~cohort.outcome_known, "success_binary"].notna().any():
        raise ValueError("Неизвестным исходам назначены метки")
    df = cohort.loc[cohort.outcome_known].sort_values("id").reset_index(drop=True)
    if not df.success_binary.eq(df.status_followup.isin(config.SUCCESS_STATUSES).astype(int)).all():
        raise ValueError("Несогласованные целевые метки")
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dates": [config.BASELINE_DATE, config.FOLLOWUP_DATE],
        "followup_date_status": "August 2026 release: author reports download in mid-August and monthly Kaggle updates on day 1; 2026-08-01 denotes release, not verified row collection timestamps",
        "n_cohort": len(cohort), "n_known": len(df), "positive": int(df.success_binary.sum()),
        "excluded_unknown": int((~cohort.outcome_known).sum()),
        "features": FEATURE_SETS, "grids": {name: parameter_grid(name) for name in FEATURE_SETS},
        "scheme_seeds": SCHEME_SEEDS, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "grouping": "128 random-order whole-batch greedy assignments; choose minimum normalized size/event imbalance, without model scores",
        "selection_metric": {"default": "fold-size-weighted inner log loss", **{m: "fold-size-weighted inner AP" for m in RANKING_MODELS}},
        "aggregation": "AP/AUC and losses weighted by fold size; P/R/Lift from summed selection counts per repeat; repeats equally weighted",
        "catboost_fixed": {"learning_rate": .03, "l2_leaf_reg": 5, "thread_count": 2},
        "comparisons": COMPARISONS,
        "uncertainty": "2000 paired resamples of companies AND batches with fixed predictions averaged by company across repeats; conditional, not full procedure confidence intervals",
        "shap": "cb_team, grouped repeat 0, held-out objects only; raw log odds",
        "tag_explanation": "lr_tags coefficients from 25 outer grouped fits; empirical 10-90% range, not confidence intervals; most frequent tags chosen without outcomes",
        "sensitivity": {"without_ik12": "exclude baseline batch IK12", "acquisitions_only": "exclude seven followup public companies, retain active/inactive/acquired"},
        "scope": "retrospective exploratory internal validation after full-cohort EDA; no future-period validation",
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost"]},
        "python": platform.python_version(),
    }
    save_json(protocol, OUT / "protocol.json")
    with threadpool_limits(limits=1):
        run_experiment(df, OUT, list(FEATURE_SETS), SCHEME_SEEDS, interpret=True)
        for name, mask in [("without_ik12", df.batch.ne("IK12")),
                           ("acquisitions_only", df.status_followup.ne("public"))]:
            run_experiment(df.loc[mask].reset_index(drop=True), OUT / name, SENSITIVITY_MODELS,
                           {"grouped": SCHEME_SEEDS["grouped"]})
    save_json({"completed_at": datetime.now(timezone.utc).isoformat(), "code_sha256": protocol["code_sha256"],
               "experiments": ["main", "without_ik12", "acquisitions_only"]}, OUT / "suite_complete.json")
    logger.info("Все основные и дополнительные эксперименты завершены")


if __name__ == "__main__":
    main()
