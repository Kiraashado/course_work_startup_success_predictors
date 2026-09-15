"""Подготовка исходных срезов и когорты активных компаний на 13.07.2023."""

import ast
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import config
from utils.logger import setup_logger

logger = setup_logger("preprocessing", log_file=config.ARTIFACTS_DIR / "preprocessing.log")
KNOWN_STATUSES = {"active", "inactive", "acquired", "public"}
BASELINE_FEATURES = [
    "team_size", "num_founders", "company_age", "batch_year", "batch_season",
    "country", "tags", "tag_count", "short_description", "description_length",
]


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def parse_tags(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    elif value is None or pd.isna(value) or not str(value).strip():
        return []
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            parsed = str(value).split(",")
        if not isinstance(parsed, (list, tuple)):
            parsed = [parsed]
    return sorted({str(tag).strip() for tag in parsed if str(tag).strip()})


def parse_batch(value):
    """Возвращает год, сезон и принадлежность к стандартному обозначению набора."""
    if value is None or pd.isna(value):
        return None, None, False
    value = str(value).strip().upper()
    match = re.fullmatch(r"(WINTER|SUMMER|SPRING|FALL)\s+(\d{4})", value)
    if match:
        return int(match[2]), match[1].title(), True
    match = re.fullmatch(r"(W|S|F|SP)(\d{2})", value)
    if match:
        return 2000 + int(match[2]), {"W": "Winter", "S": "Summer", "F": "Fall", "SP": "Spring"}[match[1]], True
    match = re.fullmatch(r"IK(\d{2})", value)
    if match:
        return 2000 + int(match[1]), None, False
    return None, None, False


def frame_state(df, id_column):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "unique_ids": int(df[id_column].nunique()),
        "duplicate_ids": int(df[id_column].duplicated().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {key: int(value) for key, value in df.isna().sum().items()},
        "status_counts": {str(key): int(value) for key, value in df["status"].value_counts(dropna=False).items()},
    }


def clean_snapshot(raw, source):
    date = config.SNAPSHOT_DATES[source]
    raw_id = "company_id" if "company_id" in raw else "id"
    report = {"before": frame_state(raw, raw_id)}
    df = raw.drop_duplicates().rename(columns=config.COLUMN_MAPPINGS[source]).copy()
    if df.columns.duplicated().any():
        raise ValueError(f"{source}: повторяющиеся имена полей")
    ids = pd.to_numeric(df["id"], errors="coerce")
    if ids.isna().any() or (ids % 1 != 0).any():
        raise ValueError(f"{source}: отсутствующий или некорректный идентификатор")
    df["id"] = ids.astype("int64")
    if not df["id"].is_unique:
        conflicts = df.loc[df["id"].duplicated(keep=False), "id"].unique().tolist()
        raise ValueError(f"{source}: несовместимые записи одного id: {conflicts[:10]}")
    df = df.reindex(columns=config.UNIFIED_SCHEMA)
    for col in df.columns:
        if col not in {"id", "tags"} and df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].replace({"": np.nan})
    df["status"] = df["status"].astype("string").str.lower().fillna("unknown")
    df.loc[~df["status"].isin(KNOWN_STATUSES), "status"] = "unknown"
    df["tags"] = df["tags"].map(parse_tags)
    df["tag_count"] = df["tags"].map(len)
    df["description_length"] = df["short_description"].astype("string").str.len().astype("Float64")
    invalid = {}
    for col in ["team_size", "num_founders", "year_founded"]:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ((values < 0) | (values % 1 != 0))
        if col == "num_founders":
            bad |= values.eq(0)
        if col == "year_founded":
            bad |= values.eq(0) | (values > int(date[:4]))
        invalid[col] = int(bad.sum())
        df[col] = values.mask(bad).astype("Float64")
    # Год основания известен без месяца и дня, поэтому возраст измеряется разностью лет.
    df["company_age"] = int(date[:4]) - df["year_founded"]
    parsed = df["batch"].map(parse_batch)
    df["batch_year"] = pd.array([item[0] for item in parsed], dtype="Int64")
    df["batch_season"] = pd.array([item[1] for item in parsed], dtype="string")
    df["regular_batch"] = [item[2] for item in parsed]
    df["future_batch"] = df["batch_year"].gt(int(date[:4])).fillna(False)
    invalid["future_batch_year"] = int(df["future_batch"].sum())
    df.loc[df["future_batch"], "batch_year"] = pd.NA
    # Страна берётся только из поля исходного среза, без географических догадок.
    df["country"] = df["country"].astype("string").str.upper().replace(
        {"USA": "US", "UNITED STATES": "US", "UK": "GB", "UNITED KINGDOM": "GB"}
    )
    df["snapshot_date"] = date
    df["source"] = source
    df = df.sort_values("id").reset_index(drop=True)
    report.update({
        "after_rows": len(df),
        "removed_exact_duplicates": len(raw) - len(df),
        "invalid_numeric_to_missing": invalid,
        "zero_team_size_retained": int(df["team_size"].eq(0).sum()),
        "empty_tag_lists": int(df["tag_count"].eq(0).sum()),
        "snapshot_date": date,
    })
    return df, report


def build_cohort(baseline, followup):
    """Левое соединение; из позднего среза переносится только статус."""
    if not baseline["id"].is_unique or not followup["id"].is_unique:
        raise ValueError("Сопоставление требует уникальных id")
    cohort = baseline.loc[baseline["status"].eq("active")].copy()
    cohort = cohort.rename(columns={"status": "status_baseline"})
    late = followup[["id", "status"]].rename(columns={"status": "status_followup"})
    cohort = cohort.merge(late, on="id", how="left", validate="one_to_one", indicator=True)
    cohort["followup_found"] = cohort["_merge"].eq("both")
    cohort["outcome_known"] = cohort["status_followup"].isin(KNOWN_STATUSES)
    cohort["status_followup"] = cohort["status_followup"].fillna("unknown")
    cohort["success_binary"] = pd.Series(pd.NA, index=cohort.index, dtype="Int64")
    known = cohort["outcome_known"]
    cohort.loc[known, "success_binary"] = (
        cohort.loc[known, "status_followup"].isin(config.SUCCESS_STATUSES).astype("int64")
    )
    cohort["followup_date"] = config.FOLLOWUP_DATE
    cohort = cohort.drop(columns="_merge").sort_values("id").reset_index(drop=True)
    if len(cohort) != int(baseline["status"].eq("active").sum()):
        raise AssertionError("Изменился состав исходной когорты")
    return cohort


def consistency_report(february, july):
    paired = february.merge(july, on="id", suffixes=("_feb", "_jul"), validate="one_to_one")
    changes = {}
    for col in ["status", "batch", "team_size", "num_founders", "year_founded", "country"]:
        left, right = paired[f"{col}_feb"], paired[f"{col}_jul"]
        same = (left.isna() & right.isna()) | left.eq(right).fillna(False)
        changes[col] = int((~same).sum())
    return {
        "common_ids": len(paired),
        "only_february": len(set(february.id) - set(july.id)),
        "only_july": len(set(july.id) - set(february.id)),
        "different_values_including_missing": changes,
        "used_to_fill_july": False,
    }


def main():
    logger.info("Подготовка когорты %s - %s", config.BASELINE_DATE, config.FOLLOWUP_DATE)
    snapshots, sources = {}, {}
    for name, filename in config.SOURCE_FILES.items():
        path = config.RAW_DATA_DIR / filename
        raw = pd.read_csv(path, low_memory=False)
        clean, report = clean_snapshot(raw, name)
        report["file"] = filename
        report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshots[name], sources[name] = clean, report
        logger.info("%s: строк %d, компаний %d, повторов %d", filename, len(raw), len(clean), len(raw) - len(clean))
    baseline, followup = snapshots["yc_2023_jul"], snapshots["yc_2026"]
    cohort = build_cohort(baseline, followup)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_date": config.BASELINE_DATE,
        "followup_date": config.FOLLOWUP_DATE,
        "sources": sources,
        "february_check": consistency_report(snapshots["yc_2023_feb"], baseline),
        "cohort": {
            "july_companies": len(baseline),
            "july_status_counts": baseline["status"].value_counts().to_dict(),
            "baseline_active": len(cohort),
            "followup_found": int(cohort["followup_found"].sum()),
            "followup_missing": int((~cohort["followup_found"]).sum()),
            "unknown_status_in_matched": int((cohort["followup_found"] & ~cohort["outcome_known"]).sum()),
            "followup_status_counts": cohort["status_followup"].value_counts().to_dict(),
            "followup_only_ids": len(set(followup.id) - set(baseline.id)),
            "missing_followup_by_batch": cohort.loc[~cohort["followup_found"], "batch"].value_counts().to_dict(),
            "missing_features": {col: int(cohort[col].isna().sum()) for col in BASELINE_FEATURES if col != "tags"},
            "no_listed_tags": int(cohort["tag_count"].eq(0).sum()),
            "zero_team_size": int(cohort["team_size"].eq(0).sum()),
        },
        "rules": {
            "population": "active in July 2023, including unmatched follow-up records",
            "positive": ["acquired", "public"],
            "negative": ["active", "inactive"],
            "unknown": "not matched or unrecognized status; never coded as zero",
            "feature_source": "July 2023 only; February used for consistency checks only",
            "founders_zero": "missing number of founders, not a company without founders",
            "company_age": "2023 minus year founded; approximate age in calendar years",
            "legacy": "dataset_a and dataset_b artifacts are not inputs to the current analysis",
        },
    }
    for name, df in {"baseline_snapshot": baseline, "followup_snapshot": followup, "analysis_cohort": cohort}.items():
        df.to_parquet(config.PREPROCESSED_DIR / f"{name}.parquet", index=False)
    save_json(report, config.PREPROCESSED_DIR / "cohort_report.json")
    logger.info("Когорта: %d компаний, известный статус %d, неизвестный %d",
                len(cohort), cohort["outcome_known"].sum(), (~cohort["outcome_known"]).sum())


if __name__ == "__main__":
    main()
