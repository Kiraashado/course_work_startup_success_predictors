"""
Утилиты для оценки качества данных.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import json
from pathlib import Path


def _make_hashable_series(series: pd.Series) -> pd.Series:
    """
    Преобразует элементы серии в хэшируемый тип.
    Списки, словари, множества → строки.
    """
    if series.dtype != object:
        return series
    
    sample = series.dropna()
    if len(sample) == 0:
        return series
    
    first_val = sample.iloc[0]
    if isinstance(first_val, (list, dict, set, tuple)):
        return series.apply(
            lambda x: str(x) if isinstance(x, (list, dict, set, tuple)) else x
        )
    return series


def assess_data_quality(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    """
    Проводит комплексную оценку качества датасета.
    Безопасна для колонок, содержащих списки/словари.
    """
    report = {
        "dataset_name": name,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "duplicates": {},
        "missing_values": {},
        "data_types": {},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }
    
    # ============================================
    # Создаём хэшируемую копию для операций, требующих хэширования
    # ============================================
    df_hash = df.copy()
    for col in df_hash.columns:
        df_hash[col] = _make_hashable_series(df_hash[col])
    
    # --- Дубликаты ---
    try:
        report["duplicates"]["full_row_duplicates"] = int(df_hash.duplicated().sum())
    except TypeError as e:
        report["duplicates"]["full_row_duplicates"] = f"error: {e}"
    
    if "id" in df.columns:
        try:
            report["duplicates"]["id_duplicates"] = int(df["id"].duplicated().sum())
        except TypeError:
            report["duplicates"]["id_duplicates"] = "error: unhashable"
    else:
        report["duplicates"]["id_duplicates"] = None
    
    # --- Пропуски (безопасно для любого типа) ---
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)
        report["missing_values"][col] = {
            "count": missing_count,
            "percentage": missing_pct,
        }
    
    # --- Типы данных ---
    for col in df.columns:
        dtype = str(df[col].dtype)
        
        # nunique на хэшируемой копии
        try:
            nunique = int(df_hash[col].nunique())
        except TypeError:
            nunique = -1  # маркер ошибки
        
        # Безопасное извлечение примеров
        sample_values = []
        try:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                for val in non_null.head(3):
                    if isinstance(val, (list, dict, set)):
                        sample_values.append(str(val))
                    elif isinstance(val, (np.integer, np.int64)):
                        sample_values.append(int(val))
                    elif isinstance(val, (np.floating, np.float64)):
                        sample_values.append(float(val))
                    else:
                        sample_values.append(val)
        except Exception:
            pass
        
        report["data_types"][col] = {
            "dtype": dtype,
            "nunique": nunique,
            "sample_values": sample_values,
        }
    
    return report


def save_quality_report(report: Dict[str, Any], output_path: Path):
    """Сохраняет отчёт о качестве в JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (set, frozenset)):
            return [make_serializable(x) for x in obj]
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj
    
    report_serializable = make_serializable(report)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False, default=str)