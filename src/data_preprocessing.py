"""
Этап 1: Предобработка данных
============================

Задачи:
1. Загрузка сырых CSV-файлов из data/raw/
2. Консолидация двух срезов 2023 года (df1 + df2 → unified_2023)
3. Подготовка датасета 2025 года (df3)
4. Унификация схемы и нормализация значений
5. Создание двух датасетов:
   - Dataset A: cross-sectional (основной, 2025, дополненный из 2023)
   - Dataset B: longitudinal/panel (компании, присутствующие в 2023 и 2025)
6. Создание бинарной целевой переменной и дельта-признаков
7. Генерация отчёта о качестве данных

Выходные артефакты (в artifacts/01_preprocessed/):
- unified_2023.parquet     — консолидированный датасет 2023
- unified_2025.parquet     — датасет 2025
- dataset_a.parquet        — основной cross-sectional датасет
- dataset_b.parquet        — longitudinal датасет (динамика 2023→2025)
- quality_report.json      — отчёт о качестве
- preprocessing_report.json — исходное состояние и изменения данных

Использование:
    python src/data_preprocessing.py
"""
import re
import ast
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

from config import config
from utils.logger import setup_logger
from utils.data_quality import assess_data_quality, save_quality_report


logger = setup_logger(
    "preprocessing",
    log_file=config.ARTIFACTS_DIR / "preprocessing.log"
)

def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Загружает все исходные CSV-файлы из data/raw/."""
    
    loaded = {}
    
    for source_name, filename in config.SOURCE_FILES.items():
        filepath = config.RAW_DATA_DIR / filename
        
        if not filepath.exists():
            continue
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df["source"] = source_name
            df["snapshot_date"] = config.SNAPSHOT_DATES.get(source_name, "unknown")
            if df.columns.duplicated().any():
                dup = df.columns[df.columns.duplicated()].tolist()
            
            loaded[source_name] = df
            
        except Exception as e:
            logger.warning("Не удалось загрузить %s: %s", filepath.name, e)
    
    if not loaded:
        raise FileNotFoundError(
            f"Не удалось загрузить ни один файл из {config.RAW_DATA_DIR}. "
            f"Ожидаемые файлы: {list(config.SOURCE_FILES.values())}"
        )
    
    return loaded


def consolidate_2023(loaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Объединяет два среза 2023 года (февраль и июль) в единый датасет.
    """
    
    df1 = loaded.get("yc_2023_feb")
    df2 = loaded.get("yc_2023_jul")
    
    if df1 is None and df2 is None:
        return pd.DataFrame()
    mapping_1 = config.COLUMN_MAPPINGS.get("yc_2023_feb", {})
    mapping_2 = config.COLUMN_MAPPINGS.get("yc_2023_jul", {})
    
    if df1 is not None:
        df1 = df1.rename(columns=mapping_1)
    
    if df2 is not None:
        df2 = df2.rename(columns=mapping_2)
    
    
    if df2 is not None:
        before = len(df2)
        df2 = df2.drop_duplicates()
    
    if df1 is None:
        return df2
    if df2 is None:
        return df1
    
    ids_old = set(df1["id"])
    ids_new = set(df2["id"])
    
    
    merged = pd.merge(
        df1, df2, on="id", how="outer",
        suffixes=("_old", "_new"), indicator=True
    )
    
    compare_cols = [c for c in config.COMPARE_COLS_2023 if f"{c}_old" in merged.columns]
    
    changes = pd.DataFrame(index=merged.index)
    for col in compare_cols:
        old_vals = merged[f"{col}_old"].fillna("__MISSING__")
        new_vals = merged[f"{col}_new"].fillna("__MISSING__")
        changes[f"{col}_changed"] = (
            (old_vals != new_vals) & merged["_merge"].eq("both")
        )
    
    changes["has_changes"] = changes.any(axis=1)
    
    unified_cols = {}
    for col in config.UNIFIED_SCHEMA:
        col_old = f"{col}_old"
        col_new = f"{col}_new"
        
        if col_old in merged.columns and col_new in merged.columns:
            unified_cols[col] = merged[col_new].combine_first(merged[col_old])
        elif col_new in merged.columns:
            unified_cols[col] = merged[col_new]
        elif col_old in merged.columns:
            unified_cols[col] = merged[col_old]
    
    unified = pd.DataFrame(unified_cols)
    unified["id"] = merged["id"]
    unified["source"] = "yc_2023_consolidated"
    unified["snapshot_date"] = "2023-07-13"
    
    for col in compare_cols:
        unified[f"{col}_changed_feb_jul"] = changes[f"{col}_changed"].values
    
    unified = unified.sort_values("id").reset_index(drop=True)
    
    assert unified["id"].is_unique, "Остались дубликаты id в unified_2023!"
    
    _save_parquet(unified, "unified_2023.parquet")
    
    return unified

def prepare_2025(loaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Подготавливает датасет 2025 года."""
    
    df3 = loaded.get("yc_2025")
    
    if df3 is None:
        return pd.DataFrame()
    
    cols_to_drop = [
        "app_video_public", "demo_day_video_public",
        "app_answers", "question_answers"
    ]
    existing = [c for c in cols_to_drop if c in df3.columns]
    df3 = df3.drop(columns=existing)
    
    mapping = config.COLUMN_MAPPINGS.get("yc_2025", {})
    df3 = df3.rename(columns=mapping)
    
    if df3.columns.duplicated().any():
        dup_cols = df3.columns[df3.columns.duplicated()].tolist()
        df3 = df3.loc[:, ~df3.columns.duplicated()]
    
    for col in config.UNIFIED_SCHEMA:
        if col not in df3.columns:
            df3[col] = np.nan
    meta_cols = ["source", "snapshot_date"]
    keep_cols = [c for c in config.UNIFIED_SCHEMA if c in df3.columns]
    for mc in meta_cols:
        if mc in df3.columns and mc not in keep_cols:
            keep_cols.append(mc)
    
    df3 = df3[keep_cols]
    df3 = df3.sort_values("id").reset_index(drop=True)
    
    _save_parquet(df3, "unified_2025.parquet")
    
    return df3

def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует значения: статусы, текстовые поля, теги, батчи."""
    df = df.copy()
    
    if "status" in df.columns:
        df["status"] = (
            df["status"].astype(str).str.strip().str.lower()
            .map(config.STATUS_MAPPING).fillna("unknown")
        )
    
    text_cols = ["name", "industry", "subindustry", "country", "city", "location_raw"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].str.lower() == "nan", col] = np.nan

    if "location_raw" in df.columns:
        location_parts = df["location_raw"].apply(_split_location)
        derived_city = location_parts.str[0]
        derived_country = location_parts.str[1]
        if "city" not in df.columns:
            df["city"] = derived_city
        else:
            df["city"] = df["city"].combine_first(derived_city)
        if "country" not in df.columns:
            df["country"] = derived_country
        else:
            df["country"] = df["country"].combine_first(derived_country)
        df["country"] = df["country"].replace({
            "US": "USA",
            "United States": "USA",
            "United States of America": "USA",
            "UK": "United Kingdom",
        })
    
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(_parse_tags)
    
    if "batch" in df.columns:
        df["batch_year"] = df["batch"].apply(_extract_batch_year)
    
    for col in ["team_size", "num_founders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "year_founded" in df.columns:
        df["company_age"] = 2025 - pd.to_numeric(df["year_founded"], errors="coerce")
        df.loc[df["company_age"] < 0, "company_age"] = np.nan
    
    return df


def _parse_tags(tags_val) -> list:
    """Преобразует строку с тегами в список."""
    if isinstance(tags_val, list):
        return [str(tag).strip() for tag in tags_val if str(tag).strip()]
    if pd.isna(tags_val):
        return []
    tags_str = str(tags_val).strip()
    if not tags_str or tags_str.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(tags_str)
        if isinstance(parsed, (list, tuple)):
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
    except (ValueError, SyntaxError):
        pass
    return [tag.strip().strip("'\"") for tag in tags_str.split(",") if tag.strip()]


def _extract_batch_year(batch_val) -> Optional[int]:
    """Извлекает год из `W22`, `S21`, `Winter 2022` и похожих форматов."""
    if pd.isna(batch_val):
        return None
    try:
        batch_str = str(batch_val).strip().upper()
        
        full_year = re.search(r"\b(?:WINTER|SUMMER|SPRING|FALL)\s+(\d{4})\b", batch_str)
        if full_year:
            return int(full_year.group(1))

        match = re.search(r"\b[WS](\d{2})\b", batch_str)
        if match:
            yy = int(match.group(1))
            if 0 <= yy <= 30:
                return 2000 + yy
            elif yy >= 95:
                return 1900 + yy
        
        digits = re.findall(r'\d+', batch_str)
        if digits:
            yy = int(digits[-1])
            if 0 <= yy <= 30:
                return 2000 + yy
            elif 95 <= yy <= 99:
                return 1900 + yy
    except (ValueError, TypeError):
        pass
    return None


def _split_location(location_val) -> tuple[Optional[str], Optional[str]]:
    """Возвращает city/country из первой локации в `all_locations`."""
    if pd.isna(location_val):
        return None, None
    first_location = str(location_val).split(";", maxsplit=1)[0].strip()
    if not first_location or first_location.lower() in {"nan", "remote"}:
        return None, None
    parts = [part.strip() for part in first_location.split(",") if part.strip()]
    if not parts:
        return None, None
    city = parts[0]
    country = parts[-1] if len(parts) >= 2 else None
    if country and re.fullmatch(r"[A-Z]{2}", country) and country not in {"US", "UK"}:
        country = "USA"
    return city, country

def create_datasets(
    unified_2023: pd.DataFrame,
    unified_2025: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Создаёт Dataset A и Dataset B."""
    
    ids_2023 = set(unified_2023["id"]) if not unified_2023.empty else set()
    ids_2025 = set(unified_2025["id"]) if not unified_2025.empty else set()
    
    common_ids = ids_2023 & ids_2025
    
    cols_to_add = [
        "year_founded", "num_founders", "founders_names",
        "country", "location_raw", "crunchbase_url", "linkedin_url"
    ]
    cols_available = []
    for c in cols_to_add:
        has_in_2023 = c in unified_2023.columns and unified_2023[c].notna().sum() > 0
        empty_in_2025 = c not in unified_2025.columns or unified_2025[c].notna().sum() == 0
        if has_in_2023 and empty_in_2025:
            cols_available.append(c)
    
    
    if cols_available and not unified_2023.empty:
        add_from_2023 = unified_2023[["id"] + cols_available].copy()
        add_from_2023 = add_from_2023.loc[:, ~add_from_2023.columns.duplicated()]
        
        cols_to_replace = [c for c in cols_available if c in unified_2025.columns]
        if cols_to_replace:
            unified_2025 = unified_2025.drop(columns=cols_to_replace)
        
        dataset_a = unified_2025.merge(add_from_2023, on="id", how="left")
    else:
        dataset_a = unified_2025.copy()
    
    dataset_a["has_data_in_2023"] = dataset_a["id"].isin(ids_2023)
    
    if dataset_a.columns.duplicated().any():
        dup = dataset_a.columns[dataset_a.columns.duplicated()].tolist()
        dataset_a = dataset_a.loc[:, ~dataset_a.columns.duplicated()]
    if "year_founded" in dataset_a.columns:
        dataset_a["year_founded"] = pd.to_numeric(dataset_a["year_founded"], errors="coerce")
        dataset_a["company_age"] = 2025 - dataset_a["year_founded"]
        dataset_a.loc[dataset_a["company_age"] < 0, "company_age"] = np.nan
    
    if "batch" in dataset_a.columns:
        dataset_a["batch_year"] = dataset_a["batch"].apply(_extract_batch_year)
    if not unified_2023.empty and not unified_2025.empty and len(common_ids) > 0:
        df_2023_b = unified_2023[unified_2023["id"].isin(common_ids)].copy()
        df_2023_b = df_2023_b.rename(columns={
            "status": "status_2023",
            "team_size": "team_size_2023",
            "tags": "tags_2023",
            "batch": "batch_2023",
        })
        
        cols_2025_b = ["id", "status", "team_size", "top_company",
                       "is_hiring", "industry", "stage"]
        cols_2025_b = [c for c in cols_2025_b if c in unified_2025.columns]
        
        df_2025_b = unified_2025[unified_2025["id"].isin(common_ids)][cols_2025_b].copy()
        df_2025_b = df_2025_b.rename(columns={
            "status": "status_2025",
            "team_size": "team_size_2025",
        })
        
        dataset_b = df_2023_b.merge(df_2025_b, on="id", how="inner")
        
        if dataset_b.columns.duplicated().any():
            dataset_b = dataset_b.loc[:, ~dataset_b.columns.duplicated()]
        
        dataset_b["status_changed"] = (
            dataset_b["status_2023"] != dataset_b["status_2025"]
        )
        dataset_b["team_size_changed"] = (
            dataset_b["team_size_2023"].fillna(-1)
            != dataset_b["team_size_2025"].fillna(-1)
        )
        dataset_b["team_size_delta"] = (
            dataset_b["team_size_2025"] - dataset_b["team_size_2023"]
        )
        
    else:
        dataset_b = pd.DataFrame()
    
    _save_parquet(dataset_a, "dataset_a.parquet")
    if not dataset_b.empty:
        _save_parquet(dataset_b, "dataset_b.parquet")
    
    
    return dataset_a, dataset_b

def create_target_variables(df: pd.DataFrame, status_col: str = "status") -> pd.DataFrame:
    """
    Создаёт бинарную целевую переменную:
    - success (1) = acquired / public
    - failure (0) = inactive
    - NaN = active (исключается из анализа)
    """
    df = df.copy()
    
    if status_col not in df.columns:
        return df
    
    df["success_binary"] = df[status_col].apply(
        lambda s: 1 if s in config.SUCCESS_STATUSES
        else (0 if s in config.FAILURE_STATUSES else np.nan)
    )
    df["status_label"] = df[status_col].apply(
        lambda s: "success" if s in config.SUCCESS_STATUSES
        else ("failure" if s in config.FAILURE_STATUSES else "unknown")
    )
    
    n_success = (df["success_binary"] == 1).sum()
    n_failure = (df["success_binary"] == 0).sum()
    n_unknown = df["success_binary"].isna().sum()
    
    
    return df

def finalize_and_report(
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
) -> None:
    """Генерирует итоговый отчёт о качестве."""
    
    dataset_a = dataset_a.dropna(subset=["id"])
    if not dataset_b.empty:
        dataset_b = dataset_b.dropna(subset=["id"])
    
    _save_parquet(dataset_a, "dataset_a_final.parquet")
    if not dataset_b.empty:
        _save_parquet(dataset_b, "dataset_b_final.parquet")
    
    report = {
        "dataset_a": assess_data_quality(dataset_a, "Dataset_A_2025"),
        "dataset_b": assess_data_quality(dataset_b, "Dataset_B_Longitudinal") if not dataset_b.empty else None,
    }
    
    report_path = config.PREPROCESSED_DIR / "quality_report.json"
    save_quality_report(report, report_path)


def _dataset_state(df: pd.DataFrame, id_col: str = "id") -> Dict[str, Any]:
    """Возвращает краткое описание структуры и полноты датасета."""
    data_columns = [col for col in df.columns if col not in {"source", "snapshot_date"}]
    data = df[data_columns]
    total_cells = data.shape[0] * data.shape[1]
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    state = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "total_missing": int(missing.sum()),
        "missing_percentage": round(
            float(missing.sum() / total_cells * 100) if total_cells else 0.0,
            2,
        ),
        "columns_with_missing": int(len(missing)),
        "missing_by_column": {col: int(value) for col, value in missing.items()},
        "full_row_duplicates": int(data.astype(str).duplicated().sum()),
    }
    if id_col in df.columns:
        state["unique_ids"] = int(df[id_col].nunique(dropna=True))
        state["duplicate_ids"] = int(df[id_col].dropna().duplicated().sum())
    return state


def save_preprocessing_report(
    loaded: Dict[str, pd.DataFrame],
    unified_2023: pd.DataFrame,
    unified_2025: pd.DataFrame,
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
) -> None:
    """Сохраняет сводку об исходных данных и изменениях в pipeline."""
    raw_states = {}
    for name, df in loaded.items():
        id_col = "id" if "id" in df.columns else "company_id"
        raw_states[name] = _dataset_state(df, id_col=id_col)

    ids_2023_feb = set(loaded.get("yc_2023_feb", pd.DataFrame()).get("company_id", []))
    ids_2023_jul = set(loaded.get("yc_2023_jul", pd.DataFrame()).get("company_id", []))
    change_columns = [col for col in unified_2023.columns if col.endswith("_changed_feb_jul")]

    key_columns = [
        "location_raw", "city", "country", "batch_year",
        "crunchbase_url", "year_founded", "num_founders",
    ]
    field_changes = {}
    for col in key_columns:
        before = int(unified_2025[col].notna().sum()) if col in unified_2025.columns else 0
        after = int(dataset_a[col].notna().sum()) if col in dataset_a.columns else 0
        field_changes[col] = {
            "non_null_before": before,
            "non_null_after": after,
            "added": after - before,
        }

    target_counts = {}
    if "status_label" in dataset_a.columns:
        target_counts = {
            str(label): int(count)
            for label, count in dataset_a["status_label"].value_counts().items()
        }

    panel_changes = None
    if not dataset_b.empty:
        delta = dataset_b["team_size_delta"].dropna()
        panel_changes = {
            "companies": int(len(dataset_b)),
            "status_changed": int(dataset_b["status_changed"].sum()),
            "team_size_changed": int(dataset_b["team_size_changed"].sum()),
            "team_size_delta": {
                "observations": int(len(delta)),
                "mean": round(float(delta.mean()), 2) if len(delta) else None,
                "median": round(float(delta.median()), 2) if len(delta) else None,
            },
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "raw_datasets": raw_states,
        "consolidation_2023": {
            "february_ids": int(len(ids_2023_feb)),
            "july_ids": int(len(ids_2023_jul)),
            "common_ids": int(len(ids_2023_feb & ids_2023_jul)),
            "only_february": int(len(ids_2023_feb - ids_2023_jul)),
            "only_july": int(len(ids_2023_jul - ids_2023_feb)),
            "result": _dataset_state(unified_2023),
            "changed_by_column": {
                col.removesuffix("_changed_feb_jul"): int(unified_2023[col].sum())
                for col in change_columns
            },
        },
        "prepared_2025": _dataset_state(unified_2025),
        "dataset_a": {
            "state": _dataset_state(dataset_a),
            "companies_present_in_2023": int(dataset_a["has_data_in_2023"].sum()),
            "field_completion": field_changes,
            "target_counts": target_counts,
        },
        "dataset_b": panel_changes,
    }

    report_path = config.PREPROCESSED_DIR / "preprocessing_report.json"
    save_quality_report(report, report_path)
    logger.info("Отчет о предобработке: %s", report_path)
    
def _save_parquet(df: pd.DataFrame, filename: str):
    """Сохраняет DataFrame в Parquet с защитой от дублирующихся колонок."""
    if df.empty:
        return
    
    output_path = config.PREPROCESSED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        df = df.loc[:, ~df.columns.duplicated()]
    
    df_to_save = df.copy()
    for col in df_to_save.columns:
        col_data = df_to_save[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
            df_to_save[col] = col_data
        
        if col_data.dtype == object:
            has_lists = col_data.apply(lambda x: isinstance(x, list)).any()
            if has_lists:
                df_to_save[col] = col_data.apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )
    
    df_to_save.to_parquet(output_path, index=False, engine="pyarrow")

def main():
    """Основной pipeline предобработки."""
    logger.info("Запуск предобработки")
    
    try:
        loaded = load_raw_data()
        
        unified_2023_raw = consolidate_2023(loaded)
        unified_2023 = normalize_values(unified_2023_raw) if not unified_2023_raw.empty else pd.DataFrame()
        
        unified_2025_raw = prepare_2025(loaded)
        unified_2025 = normalize_values(unified_2025_raw) if not unified_2025_raw.empty else pd.DataFrame()
        
        dataset_a, dataset_b = create_datasets(unified_2023, unified_2025)
        
        dataset_a = create_target_variables(dataset_a, "status")
        if not dataset_b.empty:
            dataset_b = create_target_variables(dataset_b, "status_2025")

        save_preprocessing_report(
            loaded, unified_2023, unified_2025, dataset_a, dataset_b
        )
        finalize_and_report(dataset_a, dataset_b)
        logger.info("Предобработка завершена: dataset_a=%s, dataset_b=%s", dataset_a.shape, dataset_b.shape)
        
    except Exception:
        logger.exception("Ошибка предобработки")
        raise


if __name__ == "__main__":
    main()
