import pandas as pd
import numpy as np
from typing import Dict, Any
import json
from pathlib import Path


def assess_data_quality(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    report = {
        "dataset_name": name,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "duplicates": {
            "full_row_duplicates": int(df.duplicated().sum()),
            "id_duplicates": int(df["id"].duplicated().sum()) if "id" in df.columns else None,
        },
        "missing_values": {},
        "data_types": {},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }
    
    # пропуски по колонкам
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)
        report["missing_values"][col] = {
            "count": missing_count,
            "percentage": missing_pct,
        }
    
    # типы данных
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = int(df[col].nunique())
        report["data_types"][col] = {
            "dtype": dtype,
            "nunique": nunique,
            "sample_values": df[col].dropna().head(3).tolist() if nunique > 0 else [],
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
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    def convert_dict(d):
        return {k: convert_dict(make_serializable(v)) if isinstance(v, dict) 
                else make_serializable(v) for k, v in d.items()}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert_dict(report), f, indent=2, ensure_ascii=False)