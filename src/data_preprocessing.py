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

Использование:
    python src/01_data_preprocessing.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from config import config
from utils.logger import setup_logger
from utils.data_quality import assess_data_quality, save_quality_report


# Настраиваем логгер
logger = setup_logger(
    "preprocessing",
    log_file=config.ARTIFACTS_DIR / "preprocessing.log"
)


# =============================================================================
# ШАГ 1: Загрузка исходных данных
# =============================================================================

def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Загружает все исходные CSV-файлы из data/raw/."""
    logger.info("=" * 60)
    logger.info("ШАГ 1: Загрузка исходных данных")
    logger.info("=" * 60)
    
    loaded = {}
    
    for source_name, filename in config.SOURCE_FILES.items():
        filepath = config.RAW_DATA_DIR / filename
        
        if not filepath.exists():
            logger.warning(f"Файл не найден: {filepath}. Пропускаем '{source_name}'.")
            continue
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df["source"] = source_name
            df["snapshot_date"] = config.SNAPSHOT_DATES.get(source_name, "unknown")
            
            logger.info(f"✓ Загружен '{source_name}': {df.shape[0]} строк, {df.shape[1]} колонок")
            loaded[source_name] = df
            
        except Exception as e:
            logger.error(f"✗ Ошибка при загрузке '{source_name}': {e}")
    
    if not loaded:
        raise FileNotFoundError(
            f"Не удалось загрузить ни один файл из {config.RAW_DATA_DIR}. "
            f"Ожидаемые файлы: {list(config.SOURCE_FILES.values())}"
        )
    
    return loaded


# =============================================================================
# ШАГ 2: Консолидация срезов 2023 года (df1 + df2 → unified_2023)
# =============================================================================

def consolidate_2023(loaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Объединяет два среза 2023 года (февраль и июль) в единый датасет.
    
    Логика:
    - Для компаний, присутствующих в обоих срезах, берём более свежую запись (июль).
    - Фиксируем флаги изменений (status_changed, team_size_changed и т.д.).
    """
    logger.info("=" * 60)
    logger.info("ШАГ 2: Консолидация срезов 2023 года")
    logger.info("=" * 60)
    
    df1 = loaded.get("yc_2023_feb")
    df2 = loaded.get("yc_2023_jul")
    
    if df1 is None and df2 is None:
        logger.warning("Нет данных за 2023 год. Пропускаем консолидацию.")
        return pd.DataFrame()
    
    # Удаляем полные дубликаты в df2 (в исходнике их было 4113)
    if df2 is not None:
        before = len(df2)
        df2 = df2.drop_duplicates()
        logger.info(f"  df2: удалено {before - len(df2)} полных дубликатов")
    
    if df1 is None:
        return df2
    if df2 is None:
        return df1
    
    # Пересечение по id
    ids_old = set(df1["id"])
    ids_new = set(df2["id"])
    
    logger.info(f"  Только в фев 2023: {len(ids_old - ids_new)}")
    logger.info(f"  Только в июл 2023: {len(ids_new - ids_old)}")
    logger.info(f"  В обоих срезах: {len(ids_old & ids_new)}")
    
    # Outer merge с приоритетом df2 (новее)
    # Используем маппинг для унификации колонок
    mapping_1 = config.COLUMN_MAPPINGS.get("yc_2023_feb", {})
    mapping_2 = config.COLUMN_MAPPINGS.get("yc_2023_jul", {})
    
    df1_mapped = df1.rename(columns=mapping_1)
    df2_mapped = df2.rename(columns=mapping_2)
    
    merged = pd.merge(
        df1_mapped, df2_mapped, on="id", how="outer",
        suffixes=("_old", "_new"), indicator=True
    )
    
    # Для пересекающихся колонок берём значение из df2 (new), если оно есть
    compare_cols = [c for c in config.COMPARE_COLS_2023 if f"{c}_old" in merged.columns]
    
    changes = pd.DataFrame(index=merged.index)
    for col in compare_cols:
        old_vals = merged[f"{col}_old"].fillna("__MISSING__")
        new_vals = merged[f"{col}_new"].fillna("__MISSING__")
        changes[f"{col}_changed"] = (
            (old_vals != new_vals) & merged["_merge"].eq("both")
        )
    
    changes["has_changes"] = changes.any(axis=1)
    
    logger.info(f"  Компаний с изменениями (фев→июл 2023): {changes['has_changes'].sum()}")
    
    # Формируем финальный unified_2023: приоритет new
    unified_cols = {}
    for col in config.UNIFIED_SCHEMA:
        col_old = f"{col}_old"
        col_new = f"{col}_new"
        
        if col_old in merged.columns and col_new in merged.columns:
            # Берём new, если не NaN; иначе old
            unified_cols[col] = merged[col_new].combine_first(merged[col_old])
        elif col_new in merged.columns:
            unified_cols[col] = merged[col_new]
        elif col_old in merged.columns:
            unified_cols[col] = merged[col_old]
    
    unified = pd.DataFrame(unified_cols)
    unified["id"] = merged["id"]
    unified["source"] = "yc_2023_consolidated"
    unified["snapshot_date"] = "2023-07-13"
    
    # Добавляем флаги изменений
    for col in compare_cols:
        unified[f"{col}_changed_feb_jul"] = changes[f"{col}_changed"].values
    
    unified = unified.sort_values("id").reset_index(drop=True)
    
    assert unified["id"].is_unique, "Остались дубликаты id в unified_2023!"
    
    _save_parquet(unified, "unified_2023.parquet")
    logger.info(f"Сохранён unified_2023.parquet: {unified.shape}")
    
    return unified


# =============================================================================
# ШАГ 3: Подготовка датасета 2025 года
# =============================================================================

def prepare_2025(loaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Подготавливает датасет 2025 года: унифицирует схему, удаляет ненужные колонки."""
    logger.info("=" * 60)
    logger.info("ШАГ 3: Подготовка датасета 2025 года")
    logger.info("=" * 60)
    
    df3 = loaded.get("yc_2025")
    
    if df3 is None:
        logger.warning("Нет данных за 2025 год.")
        return pd.DataFrame()
    
    # Удаляем служебные колонки
    cols_to_drop = [
        "app_video_public", "demo_day_video_public",
        "app_answers", "question_answers"
    ]
    existing = [c for c in cols_to_drop if c in df3.columns]
    df3 = df3.drop(columns=existing)
    
    # Применяем маппинг
    mapping = config.COLUMN_MAPPINGS.get("yc_2025", {})
    df3 = df3.rename(columns=mapping)
    
    # Добавляем недостающие колонки из UNIFIED_SCHEMA
    for col in config.UNIFIED_SCHEMA:
        if col not in df3.columns:
            df3[col] = np.nan
    
    # Оставляем только нужные колонки + метаданные
    meta_cols = ["source", "snapshot_date"]
    keep_cols = [c for c in config.UNIFIED_SCHEMA if c in df3.columns] + meta_cols
    df3 = df3[keep_cols]
    
    df3 = df3.sort_values("id").reset_index(drop=True)
    
    _save_parquet(df3, "unified_2025.parquet")
    logger.info(f"Сохранён unified_2025.parquet: {df3.shape}")
    
    return df3


# =============================================================================
# ШАГ 4: Нормализация значений
# =============================================================================

def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует значения: статусы, текстовые поля, теги, батчи."""
    df = df.copy()
    
    # Нормализация статусов
    if "status" in df.columns:
        df["status"] = (
            df["status"].astype(str).str.strip().str.lower()
            .map(config.STATUS_MAPPING).fillna("unknown")
        )
    
    # Текстовые поля
    text_cols = ["name", "industry", "subindustry", "country", "city"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].str.lower() == "nan", col] = np.nan
    
    # Теги → списки
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(_parse_tags)
    
    # Год из батча
    if "batch" in df.columns:
        df["batch_year"] = df["batch"].apply(_extract_batch_year)
    
    # Числовые поля
    for col in ["team_size", "num_founders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # company_age (если есть year_founded)
    if "year_founded" in df.columns:
        df["company_age"] = 2025 - pd.to_numeric(df["year_founded"], errors="coerce")
        df.loc[df["company_age"] < 0, "company_age"] = np.nan
    
    return df


def _parse_tags(tags_val) -> list:
    """Преобразует строку с тегами в список."""
    if pd.isna(tags_val):
        return []
    if isinstance(tags_val, list):
        return tags_val
    tags_str = str(tags_val).strip("[]'\"")
    if not tags_str or tags_str == "nan":
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def _extract_batch_year(batch_val) -> Optional[int]:
    """Извлекает год из батча вида 'W22', 'S21' и т.д."""
    if pd.isna(batch_val):
        return None
    try:
        batch_str = str(batch_val).strip()
        if len(batch_str) >= 3:
            yy = int(batch_str[-2:])
            return 2000 + yy if yy < 50 else 1900 + yy
    except (ValueError, TypeError):
        pass
    return None


# =============================================================================
# ШАГ 5: Создание Dataset A и Dataset B
# =============================================================================

def create_datasets(
    unified_2023: pd.DataFrame,
    unified_2025: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Создаёт два датасета:
    - Dataset A: cross-sectional (основной, 2025 + данные из 2023)
    - Dataset B: longitudinal (компании в обоих срезах, 2023 и 2025)
    """
    logger.info("=" * 60)
    logger.info("ШАГ 5: Создание Dataset A и Dataset B")
    logger.info("=" * 60)
    
    # --- Пересечение id ---
    ids_2023 = set(unified_2023["id"]) if not unified_2023.empty else set()
    ids_2025 = set(unified_2025["id"]) if not unified_2025.empty else set()
    
    common_ids = ids_2023 & ids_2025
    only_2023 = ids_2023 - ids_2025
    only_2025 = ids_2025 - ids_2023
    
    logger.info(f"  Уникальных id в 2023: {len(ids_2023)}")
    logger.info(f"  Уникальных id в 2025: {len(ids_2025)}")
    logger.info(f"  Общих: {len(common_ids)}")
    logger.info(f"  Только в 2023: {len(only_2023)}")
    logger.info(f"  Только в 2025: {len(only_2025)}")
    
    # --- Dataset A (cross-sectional) ---
    # Основа — 2025. Дополняем колонками из 2023, которых нет в 2025.
    cols_to_add = [
        "year_founded", "num_founders", "founders_names",
        "country", "location_raw", "crunchbase_url", "linkedin_url"
    ]
    cols_available = [
        c for c in cols_to_add
        if c in unified_2023.columns and c not in unified_2025.columns
    ]
    
    if cols_available and not unified_2023.empty:
        add_from_2023 = unified_2023[["id"] + cols_available]
        dataset_a = unified_2025.merge(add_from_2023, on="id", how="left")
        logger.info(f"  Dataset A: дополнен {len(cols_available)} колонками из 2023")
    else:
        dataset_a = unified_2025.copy()
    
    dataset_a["has_data_in_2023"] = dataset_a["id"].isin(ids_2023)
    
    # --- Dataset B (longitudinal) ---
    # Только компании, присутствующие в обоих срезах
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
        
        # Флаги изменений
        dataset_b["status_changed"] = (
            dataset_b["status_2023"] != dataset_b["status_2025"]
        )
        dataset_b["team_size_changed"] = (
            dataset_b["team_size_2023"].fillna(-1)
            != dataset_b["team_size_2025"].fillna(-1)
        )
        
        # Дельта team_size
        dataset_b["team_size_delta"] = (
            dataset_b["team_size_2025"] - dataset_b["team_size_2023"]
        )
        
        logger.info(f"  Dataset B: {dataset_b.shape}")
        logger.info(f"  Изменивших статус: {dataset_b['status_changed'].sum()}")
    else:
        dataset_b = pd.DataFrame()
        logger.warning("  Dataset B не создан: нет пересечения между срезами")
    
    _save_parquet(dataset_a, "dataset_a.parquet")
    _save_parquet(dataset_b, "dataset_b.parquet")
    
    logger.info(f"Сохранён dataset_a.parquet: {dataset_a.shape}")
    logger.info(f"Сохранён dataset_b.parquet: {dataset_b.shape}")
    
    return dataset_a, dataset_b


# =============================================================================
# ШАГ 6: Создание бинарной целевой переменной
# =============================================================================

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
    
    logger.info(f"  Целевая ({status_col}): success={n_success}, failure={n_failure}, unknown={n_unknown}")
    
    return df


# =============================================================================
# ШАГ 7: Финальная очистка и отчёты
# =============================================================================

def finalize_and_report(
    dataset_a: pd.DataFrame,
    dataset_b: pd.DataFrame,
) -> None:
    """Генерирует итоговый отчёт о качестве."""
    logger.info("=" * 60)
    logger.info("ШАГ 7: Финальная очистка и генерация отчётов")
    logger.info("=" * 60)
    
    # Удаляем строки без id
    dataset_a = dataset_a.dropna(subset=["id"])
    if not dataset_b.empty:
        dataset_b = dataset_b.dropna(subset=["id"])
    
    # Сохраняем финальные версии
    _save_parquet(dataset_a, "dataset_a_final.parquet")
    if not dataset_b.empty:
        _save_parquet(dataset_b, "dataset_b_final.parquet")
    
    # Отчёт о качестве
    report = {
        "dataset_a": assess_data_quality(dataset_a, "Dataset_A_2025"),
        "dataset_b": assess_data_quality(dataset_b, "Dataset_B_Longitudinal") if not dataset_b.empty else None,
    }
    
    report_path = config.PREPROCESSED_DIR / "quality_report.json"
    save_quality_report(report, report_path)
    logger.info(f"Сохранён отчёт: {report_path}")
    
    # Сводка
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " ИТОГОВАЯ СВОДКА ".center(58) + "║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info(f"║  Dataset A (2025): {dataset_a.shape[0]:>6} записей, {dataset_a.shape[1]:>3} колонок     ║")
    if not dataset_b.empty:
        logger.info(f"║  Dataset B (panel): {dataset_b.shape[0]:>5} записей, {dataset_b.shape[1]:>3} колонок     ║")
    
    n_sf_a = dataset_a["success_binary"].notna().sum()
    logger.info(f"║  Dataset A: для SF-анализа {n_sf_a:>6} компаний                  ║")
    logger.info("╚" + "═" * 58 + "╝")


# =============================================================================
# Вспомогательные функции
# =============================================================================

def _save_parquet(df: pd.DataFrame, filename: str):
    """Сохраняет DataFrame в Parquet."""
    if df.empty:
        logger.warning(f"Попытка сохранить пустой DataFrame как {filename}")
        return
    
    output_path = config.PREPROCESSED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_to_save = df.copy()
    for col in df_to_save.columns:
        if df_to_save[col].dtype == object:
            has_lists = df_to_save[col].apply(lambda x: isinstance(x, list)).any()
            if has_lists:
                df_to_save[col] = df_to_save[col].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )
    
    df_to_save.to_parquet(output_path, index=False, engine="pyarrow")


# =============================================================================
# Главная функция
# =============================================================================

def main():
    """Основной pipeline предобработки."""
    logger.info("Начало выполнения pipeline предобработки данных")
    logger.info(f"Корневая директория проекта: {config.PROJECT_ROOT}")
    
    try:
        loaded = load_raw_data()
        
        # Консолидация 2023
        unified_2023_raw = consolidate_2023(loaded)
        unified_2023 = normalize_values(unified_2023_raw) if not unified_2023_raw.empty else pd.DataFrame()
        
        # Подготовка 2025
        unified_2025_raw = prepare_2025(loaded)
        unified_2025 = normalize_values(unified_2025_raw) if not unified_2025_raw.empty else pd.DataFrame()
        
        # Создание Dataset A и Dataset B
        dataset_a, dataset_b = create_datasets(unified_2023, unified_2025)
        
        # Целевые переменные
        dataset_a = create_target_variables(dataset_a, "status")
        if not dataset_b.empty:
            dataset_b = create_target_variables(dataset_b, "status_2025")
        
        # Финализация
        finalize_and_report(dataset_a, dataset_b)
        
        logger.info("✓ Pipeline предобработки успешно завершён!")
        
    except Exception as e:
        logger.error(f"✗ Pipeline завершился с ошибкой: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()