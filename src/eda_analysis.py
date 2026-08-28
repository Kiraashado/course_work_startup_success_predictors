"""
Этап 2: Exploratory Data Analysis (EDA)
========================================

Базовый обзор данных: статистика, распределения, пропуски, визуализация.

Входные артефакты:
- artifacts/01_preprocessed/dataset_a_final.parquet
- artifacts/01_preprocessed/dataset_b_final.parquet

Выходные артефакты:
- artifacts/eda/*.png — графики
- artifacts/eda/eda_summary.json — сводная статистика

Использование:
    python src/eda_analysis.py
"""

import json
import ast
from collections import Counter
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from config import config
from utils.logger import setup_logger

FIGURE_DPI = 150
DEFAULT_FIGSIZE = (10, 6)
COLORS = {
    "primary": "#4C72B0",
    "dark": "#24476F",
    "medium": "#6F91C5",
    "soft": "#9DB7DC",
    "muted": "#C4D3E8",
    "light": "#EAF0F8",
    "text": "#2F2F2F",
}
STATUS_COLORS = {
    "active": COLORS["primary"],
    "inactive": COLORS["dark"],
    "acquired": COLORS["medium"],
    "public": COLORS["soft"],
    "unknown": COLORS["muted"],
}
SF_COLORS = {
    "success": COLORS["primary"],
    "failure": COLORS["dark"],
}
CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "blue_diverging",
    [COLORS["dark"], COLORS["light"], COLORS["primary"]],
)

sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette=[COLORS["primary"], COLORS["medium"], COLORS["soft"]],
    font="DejaVu Sans",
    font_scale=1.0,
    rc={
        "axes.edgecolor": "#D5D5D5",
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "grid.alpha": 0.25,
        "grid.color": "#BDBDBD",
        "legend.frameon": False,
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
    },
)
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI,
    "figure.figsize": DEFAULT_FIGSIZE,
    "axes.unicode_minus": False,
})

logger = setup_logger(
    "eda",
    log_file=config.ARTIFACTS_DIR / "eda.log"
)

def load_datasets() -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Загружает датасеты A и B."""
    
    path_a = config.PREPROCESSED_DIR / "dataset_a_final.parquet"
    path_b = config.PREPROCESSED_DIR / "dataset_b_final.parquet"
    
    if not path_a.exists():
        raise FileNotFoundError(
            f"Dataset A не найден: {path_a}\n"
            "Сначала запустите: python src/data_preprocessing.py"
        )
    
    dataset_a = pd.read_parquet(path_a)
    
    dataset_b = None
    if path_b.exists():
        dataset_b = pd.read_parquet(path_b)
    else:
        logger.info("Dataset B отсутствует; лонгитюдный анализ пропущен")
    
    return dataset_a, dataset_b


def save_figure(fig: Figure, name: str) -> None:
    """Сохраняет фигуру в artifacts/eda/."""
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EDA_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def style_axis(
    ax: Axes,
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    grid_axis: str = "y",
) -> None:
    """Применяет единое оформление к осям графика."""
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis)
    sns.despine(ax=ax)


def add_bar_labels(
    ax: Axes,
    bars: BarContainer,
    labels: list[str],
    *,
    horizontal: bool = False,
) -> None:
    """Добавляет одинаково оформленные подписи к столбцам."""
    ax.bar_label(
        bars,
        labels=labels,
        label_type="edge",
        padding=4,
        fontsize=9,
        fontweight="bold",
    )
    if horizontal:
        ax.margins(x=0.15)
    else:
        ax.margins(y=0.15)


def get_sf_dataset(df: pd.DataFrame, status_col: str = "status") -> pd.DataFrame:
    """
    Возвращает подвыборку для анализа success vs failure.
    Исключает active/unknown — их исход ещё неизвестен.
    """
    success_mask = df[status_col].isin(config.SUCCESS_STATUSES)
    failure_mask = df[status_col].isin(config.FAILURE_STATUSES)
    return df[success_mask | failure_mask].copy()


def binarize_status(status: str) -> str:
    """Преобразует статус в бинарную метку."""
    if status in config.SUCCESS_STATUSES:
        return "success"
    if status in config.FAILURE_STATUSES:
        return "failure"
    return "unknown"


def extract_batch_season(batch: Any) -> str:
    """Приводит сокращённые и полные названия сезонов к общему формату."""
    value = str(batch).strip().upper()
    if value.startswith(("W", "WINTER")):
        return "Winter"
    if value.startswith(("S", "SUMMER")):
        return "Summer"
    if value.startswith(("F", "FALL")):
        return "Fall"
    if value.startswith("SPRING"):
        return "Spring"
    return "Unknown"


def parse_tags(tags: Any) -> list[str]:
    """Возвращает теги из list или его строкового представления."""
    if isinstance(tags, (list, tuple, np.ndarray)):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    if pd.isna(tags) or str(tags).strip() in {"", "[]", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(str(tags))
        if isinstance(parsed, (list, tuple)):
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
    except (ValueError, SyntaxError):
        pass
    return [tag.strip() for tag in str(tags).split(",") if tag.strip()]


def analytical_industry(row: pd.Series) -> str:
    """Выделяет AI из тегов, иначе сохраняет верхнеуровневую отрасль YC."""
    normalized_tags = [tag.casefold() for tag in parse_tags(row.get("tags"))]
    tags = " | ".join(normalized_tags)
    ai_markers = ("artificial intelligence", "generative ai", "machine learning", "ai-")
    if any(marker in tags for marker in ai_markers) or "ai" in normalized_tags:
        return "Artificial Intelligence"
    industry = row.get("industry")
    return str(industry).strip() if pd.notna(industry) else "Unknown"

def analyze_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Анализирует пропуски в датасете."""
    
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    missing_pct_total = total_missing / total_cells * 100 if total_cells else 0.0
    
    
    missing_per_col = df.isnull().sum()
    missing_pct = (missing_per_col / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_pct) * 0.3)))
        colors = [
            COLORS["dark"] if value > 50
            else COLORS["medium"] if value > 20
            else COLORS["primary"]
            for value in missing_pct.values[::-1]
        ]
        bars = ax.barh(
            missing_pct.index[::-1], missing_pct.values[::-1], color=colors
        )
        style_axis(
            ax,
            title=(
                "Пропущенные значения в Dataset A (2025)\n"
                f"Всего пропущено: {missing_pct_total:.1f}% ячеек"
            ),
            xlabel="Доля пропусков, %",
        )
        ax.axvline(50, color=COLORS["dark"], linestyle="--", label="50% (критично)")
        ax.axvline(20, color=COLORS["medium"], linestyle="--", label="20% (внимание)")
        add_bar_labels(
            ax, bars, [f"{value:.1f}%" for value in missing_pct.values[::-1]],
            horizontal=True,
        )
        ax.legend(loc="lower right")
        ax.set_xlim(0, min(100, missing_pct.max() * 1.15))
        save_figure(fig, "01_missing_values")

        top_missing_cols = missing_pct.head(10).index.tolist()
        df_missing_pattern = df[top_missing_cols].isnull().astype(int)
        
        sample_size = min(500, len(df))
        df_sample = df_missing_pattern.sample(n=sample_size, random_state=42)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            df_sample.T, 
            cmap=[COLORS["light"], COLORS["dark"]],
            cbar_kws={"label": "Пропуск (1) / Наличие (0)"},
            ax=ax,
            yticklabels=True,
            xticklabels=False
        )
        style_axis(
            ax,
            title=(f"Паттерны пропусков (выборка {sample_size} строк)\n"
                   "Топ-10 колонок с наибольшим числом пропусков"),
            xlabel="Индекс строки",
            ylabel="Колонка",
            grid_axis="",
        )
        save_figure(fig, "02_missing_patterns")
    
    return {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_pct_total": round(missing_pct_total, 2),
        "columns_with_missing": len(missing_pct),
        "columns_with_missing_gt_50": int((missing_pct > 50).sum()),
        "missing_by_column": {col: round(pct, 2) for col, pct in missing_pct.items()},
    }

def analyze_target_variable(df: pd.DataFrame) -> dict[str, Any]:
    """Анализирует распределение целевой переменной."""
    
    if "status" not in df.columns:
        return {}
    
    status_counts = df["status"].value_counts()
    status_pct = df["status"].value_counts(normalize=True) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = [STATUS_COLORS.get(s, COLORS["muted"]) for s in status_counts.index]
    bars = axes[0].bar(
        status_counts.index, status_counts.values,
        color=colors
    )
    add_bar_labels(
        axes[0], bars,
        [f"{value}\n({status_pct[status]:.1f}%)"
         for status, value in status_counts.items()],
    )
    style_axis(
        axes[0], title="Количество стартапов по статусу",
        xlabel="Статус", ylabel="Количество компаний",
    )
    
    wedges, texts, autotexts = axes[1].pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.85,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
    )
    
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)
    
    axes[1].set_title("Доля стартапов по статусу", pad=12)
    
    centre_circle = plt.Circle((0, 0), 0.70, fc="white", ec="#D5D5D5")
    axes[1].add_artist(centre_circle)
    axes[1].text(0, 0, f"{len(df)}\nвсего", ha="center", va="center", 
                fontsize=14, fontweight="bold")
    
    plt.suptitle(
        "Распределение целевой переменной status (Dataset A, 2025)",
        fontsize=14, fontweight="bold", y=1.02
    )
    save_figure(fig, "03_status_distribution")
    
    df_copy = df.copy()
    df_copy["status_binary"] = df_copy["status"].apply(binarize_status)
    df_sf = get_sf_dataset(df_copy)
    
    sf_counts = df_sf["status_binary"].value_counts()
    if sf_counts.empty:
        return {
            "status_counts": {s: int(c) for s, c in status_counts.items()},
            "status_percentages": {s: round(p, 2) for s, p in status_pct.items()},
            "sf_analysis": {"success": 0, "failure": 0, "excluded": int(len(df))},
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        sf_counts.index, sf_counts.values,
        color=[SF_COLORS.get(s, COLORS["muted"]) for s in sf_counts.index],
        width=0.6,
    )
    add_bar_labels(
        ax, bars,
        [f"{value}\n({value / len(df_sf) * 100:.1f}%)" for value in sf_counts.values],
    )
    style_axis(
        ax,
        title="Распределение для бинарного анализа\n(Success vs Failure, без active)",
        xlabel="Бинарный статус",
        ylabel="Количество компаний",
    )
    save_figure(fig, "04_status_binary")
    
    return {
        "status_counts": {s: int(c) for s, c in status_counts.items()},
        "status_percentages": {s: round(p, 2) for s, p in status_pct.items()},
        "sf_analysis": {
            "success": int((df_sf["status_binary"] == "success").sum()),
            "failure": int((df_sf["status_binary"] == "failure").sum()),
            "excluded": int(len(df) - len(df_sf)),
        },
    }

def analyze_numerical_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Анализирует числовые признаки.
    Для team_size используется логарифмическая шкала из-за сильного правого скоса.
    """
    
    num_cols = [c for c in ["team_size", "num_founders", "company_age"] 
                if c in df.columns]
    
    valid_num_cols = []
    for col in num_cols:
        non_null_count = df[col].notna().sum()
        if non_null_count > 0:
            valid_num_cols.append(col)
    
    if not valid_num_cols:
        return {}
    
    desc = df[valid_num_cols].describe().round(2)
    
    skewness = {}
    for col in valid_num_cols:
        data = df[col].dropna()
        skew = data.skew()
        skewness[col] = round(float(skew), 2) if not pd.isna(skew) else 0.0
    n_cols = len(valid_num_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, valid_num_cols):
        data = df[col].dropna()
        
        use_log = abs(data.skew()) > 1.5 and (data >= 0).all()
        
        if use_log:
            log_data = np.log10(data + 1)
            
            ax.hist(log_data, bins=40, color=COLORS["primary"])
            
            median_val = data.median()
            ax.axvline(np.log10(median_val + 1), color=COLORS["dark"], linestyle="--",
                      linewidth=2, label=f"Медиана: {median_val:.0f}")
            
            ax.set_title(f"{col}\n(log₁₀(x + 1), skew={data.skew():.1f})", 
                        fontsize=11, fontweight="bold")
            ax.set_xlabel(f"log₁₀({col} + 1)")
            
            tick_vals = [0, 1, 10, 100, 1000, 10000]
            tick_labels = ["0", "1", "10", "100", "1K", "10K"]
            max_val = data.max()
            ax.set_xticks([np.log10(v + 1) for v in tick_vals if v <= max_val])
            ax.set_xticklabels([l for v, l in zip(tick_vals, tick_labels) 
                               if v <= max_val])
            
        else:
            cap = data.quantile(0.99)
            data_capped = data[data <= cap]
            
            ax.hist(data_capped, bins=40, color=COLORS["primary"])
            
            median_val = data.median()
            ax.axvline(median_val, color=COLORS["dark"], linestyle="--",
                      linewidth=2, label=f"Медиана: {median_val:.0f}")
            
            ax.set_title(f"{col}\n(skew={data.skew():.1f})",
                        fontsize=11, fontweight="bold")
            ax.set_xlabel(col)
        
        ax.set_ylabel("Количество компаний")
        ax.legend(fontsize=9, loc="upper right")
    
    plt.suptitle(
        "Распределения числовых признаков (Dataset A)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_figure(fig, "05_numerical_distributions")
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, valid_num_cols):
        data = df[col].dropna()
        use_log = abs(data.skew()) > 1.5 and (data >= 0).all()
        
        if use_log:
            data_plot = np.log10(data + 1)
            ax.boxplot(data_plot, orientation="vertical", patch_artist=True,
                           boxprops={"facecolor": COLORS["primary"], "alpha": 0.8},
                           medianprops={"color": COLORS["dark"], "linewidth": 2})
            
            ax.set_title(f"{col} (log₁₀ шкала)", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"log₁₀({col})")
            
            q1, median, q3 = data_plot.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            ax.text(0.05, 0.95, 
                   f"Q1: {10**q1 - 1:.0f}\n"
                   f"Median: {10**median - 1:.0f}\n"
                   f"Q3: {10**q3 - 1:.0f}\n"
                   f"IQR: {iqr:.2f} (log)",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": COLORS["light"], "alpha": 0.8})
            
        else:
            ax.boxplot(data, orientation="vertical", patch_artist=True,
                           boxprops={"facecolor": COLORS["primary"], "alpha": 0.8},
                           medianprops={"color": COLORS["dark"], "linewidth": 2})
            
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_ylabel(col)
            
            q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            ax.text(0.05, 0.95,
                   f"Q1: {q1:.0f}\n"
                   f"Median: {median:.0f}\n"
                   f"Q3: {q3:.0f}\n"
                   f"IQR: {iqr:.0f}",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": COLORS["light"], "alpha": 0.8})
    
    plt.suptitle(
        "Boxplot'ы числовых признаков (выбросы)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_figure(fig, "06_numerical_boxplots")
    corr_matrix = None
    if len(valid_num_cols) >= 2:
        df_corr = df[valid_num_cols].dropna()
        if len(df_corr) > 0:
            corr_matrix = df_corr.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt=".3f",
                cmap=CORRELATION_CMAP, center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, ax=ax,
                cbar_kws={"label": "Коэффициент корреляции Пирсона"}
            )
            ax.set_title("Корреляционная матрица числовых признаков",
                        fontsize=12, fontweight="bold")
            plt.tight_layout()
            save_figure(fig, "07_correlation_matrix")
            
    if 2 <= len(valid_num_cols) <= 3:
        df_pair = df[valid_num_cols].dropna()
        if len(df_pair) > 0:
            if "team_size" in df_pair.columns:
                df_pair["log_team_size"] = np.log10(df_pair["team_size"] + 1)
                df_pair = df_pair.drop(columns=["team_size"])
            
            pairplot = sns.pairplot(
                df_pair, diag_kind="kde",
                plot_kws={"alpha": 0.5, "s": 20},
                diag_kws={"alpha": 0.7}
            )
            pairplot.fig.suptitle(
                "Попарные распределения числовых признаков",
                fontsize=14, fontweight="bold", y=1.02
            )
            plt.tight_layout()
            save_figure(pairplot.fig, "08_pairplot")
    
    return {
        "descriptive_stats": desc.to_dict(),
        "skewness": skewness,
        "correlation_matrix": corr_matrix.to_dict() if corr_matrix is not None else None,
    }

def analyze_categorical_features(df: pd.DataFrame) -> dict[str, Any]:
    """Анализирует категориальные признаки."""
    
    results = {}
    if "batch_year" in df.columns and df["batch_year"].notna().sum() > 0:
        
        batch_counts = df.groupby("batch_year").size().reset_index(name="count")
        batch_counts = batch_counts.dropna(subset=["batch_year"])
        
        if len(batch_counts) == 0:
            return results
        else:
            batch_counts["batch_year"] = batch_counts["batch_year"].astype(int)
            
            
            fig, ax = plt.subplots(figsize=(14, 6))
            bars = ax.bar(
                batch_counts["batch_year"], batch_counts["count"],
                color=COLORS["primary"]
            )
            
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height + 5,
                    str(int(height)), ha="center", va="bottom",
                    fontsize=8, fontweight="bold"
                )
            
            ax.set_title(
                "Количество стартапов YC по году батча",
                fontsize=13, fontweight="bold"
            )
            ax.set_xlabel("Год батча", fontsize=11)
            ax.set_ylabel("Количество стартапов", fontsize=11)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            plt.xticks(rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, "09_batch_by_year")
            
            results["batch_by_year"] = {
                int(row["batch_year"]): int(row["count"])
                for _, row in batch_counts.iterrows()
            }
        
        if "batch" in df.columns:
            df_season = df.copy()
            df_season["season"] = df_season["batch"].apply(extract_batch_season)
            
            season_counts = df_season["season"].value_counts()
            
            if len(season_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(
                    season_counts.index, season_counts.values,
                    color=COLORS["primary"],
                )
                
                for bar, val in zip(bars, season_counts.values):
                    pct = val / season_counts.sum() * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        f"{val}\n({pct:.1f}%)",
                        ha="center", va="bottom", fontweight="bold"
                    )
                
                ax.set_title(
                    "Распределение батчей по сезонам",
                    fontsize=12, fontweight="bold"
                )
                ax.set_ylabel("Количество стартапов")
                ax.set_ylim(0, season_counts.max() * 1.2)
                
                plt.tight_layout()
                save_figure(fig, "10_batch_seasonality")
                
                results["seasonality"] = {s: int(c) for s, c in season_counts.items()}
    else:
        pass
    if "industry" in df.columns:
        
        industry_series = df.apply(analytical_industry, axis=1)
        top_industries = industry_series.value_counts().head(10)
        total_with_industry = int(industry_series.ne("Unknown").sum())
        
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.barh(
            top_industries.index[::-1], top_industries.values[::-1],
            color=COLORS["primary"]
        )
        
        for bar, val in zip(bars, top_industries.values[::-1]):
            pct = val / total_with_industry * 100
            ax.text(
                val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val} ({pct:.1f}%)", va="center", fontsize=9
            )
        
        ax.set_title(
            f"Топ-10 аналитических отраслей YC\n"
            f"(AI выделен по тегам; всего: {total_with_industry})",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Количество стартапов", fontsize=11)
        ax.set_xlim(0, top_industries.max() * 1.15)
        
        plt.tight_layout()
        save_figure(fig, "11_top_industries")
        
        results["top_industries"] = {ind: int(cnt) for ind, cnt in top_industries.items()}
    country_col = None
    if "country" in df.columns and df["country"].notna().sum() > 0:
        country_col = "country"
    elif "all_locations" in df.columns:
        df_copy = df.copy()
        df_copy["country_extracted"] = (
            df_copy["all_locations"]
            .str.split(",")
            .str[-1]
            .str.strip()
        )
        if df_copy["country_extracted"].notna().sum() > 0:
            country_col = "country_extracted"
            df = df_copy
    
    if country_col:
        
        top_countries = df[country_col].value_counts().head(10)
        total_with_country = df[country_col].notna().sum()
        
        
        if len(top_countries) == 0:
            pass
        else:
            fig, ax = plt.subplots(figsize=(11, 6))
            bars = ax.barh(
                top_countries.index[::-1], top_countries.values[::-1],
                color=COLORS["primary"]
            )
            add_bar_labels(
                ax,
                bars,
                [f"{val} ({val / total_with_country * 100:.1f}%)"
                 for val in top_countries.values[::-1]],
                horizontal=True,
            )
            
            ax.set_title(
                f"Топ-10 стран по количеству стартапов YC\n(всего с указанной страной: {total_with_country})",
                fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Количество стартапов (log-шкала)", fontsize=11)
            ax.set_xscale("log")
            
            plt.tight_layout()
            save_figure(fig, "12_top_countries")
            
            results["top_countries"] = {c: int(cnt) for c, cnt in top_countries.items()}
    else:
        pass
    if "tags" in df.columns:
        
        all_tags = []
        for tags in df["tags"]:
            all_tags.extend(parse_tags(tags))
        
        if all_tags:
            tag_counts = pd.Series(Counter(all_tags)).sort_values(ascending=False)
            top_tags = tag_counts.head(20)
            
            
            fig, ax = plt.subplots(figsize=(11, 8))
            bars = ax.barh(
                top_tags.index[::-1], top_tags.values[::-1],
                color=COLORS["primary"]
            )
            
            for bar, val in zip(bars, top_tags.values[::-1]):
                ax.text(
                    val + 5, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9
                )
            
            ax.set_title(
                f"Топ-20 тегов стартапов YC\n(всего уникальных тегов: {len(tag_counts)})",
                fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Количество стартапов с тегом", fontsize=11)
            ax.set_xlim(0, top_tags.max() * 1.1)
            
            plt.tight_layout()
            save_figure(fig, "13_top_tags")
            
            results["top_tags"] = {tag: int(cnt) for tag, cnt in top_tags.items()}
            results["total_unique_tags"] = len(tag_counts)
    
    return results

def analyze_temporal_features(df: pd.DataFrame) -> dict[str, Any]:
    """Анализирует временные характеристики."""
    
    results = {}
    if "year_founded" in df.columns and df["year_founded"].notna().sum() > 0:
        
        df_founded = df[df["year_founded"].notna()].copy()
        df_founded["year_founded"] = pd.to_numeric(
            df_founded["year_founded"], errors="coerce"
        ).astype("Int64")  # nullable integer
        
        df_founded = df_founded[df_founded["year_founded"].notna()]
        
        if len(df_founded) == 0:
            pass
        else:
            year_counts = df_founded["year_founded"].value_counts().sort_index()
            
            year_counts = year_counts[
                (year_counts.index >= 2005) & (year_counts.index <= 2025)
            ]
            
            if len(year_counts) == 0:
                pass
            else:
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.bar(
                    year_counts.index.astype(int), year_counts.values,
                    color=COLORS["primary"]
                )
                
                ax.set_title(
                    f"Распределение стартапов по году основания\n"
                    f"(медиана: {df_founded['year_founded'].median():.0f})",
                    fontsize=12, fontweight="bold"
                )
                ax.set_xlabel("Год основания", fontsize=11)
                ax.set_ylabel("Количество стартапов", fontsize=11)
                ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
                plt.xticks(rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, "14_year_founded")
                
                results["year_founded"] = {
                    "min": int(year_counts.index.min()),
                    "max": int(year_counts.index.max()),
                    "median": float(df_founded["year_founded"].median()),
                    "distribution": {int(y): int(c) for y, c in year_counts.items()},
                }
    else:
        pass
    if "company_age" in df.columns and df["company_age"].notna().sum() > 0:
        
        age_data = df["company_age"].dropna()
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cap = age_data.quantile(0.99)
        age_capped = age_data[age_data <= cap]
        
        ax.hist(
            age_capped, bins=30, color=COLORS["primary"]
        )
        
        median_age = age_data.median()
        ax.axvline(
            median_age, color=COLORS["dark"], linestyle="--",
            linewidth=2, label=f"Медиана: {median_age:.0f} лет"
        )
        
        ax.set_title(
            f"Распределение возраста компаний\n"
            f"(средний: {age_data.mean():.1f}, медиана: {median_age:.0f})",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Возраст компании (лет)", fontsize=11)
        ax.set_ylabel("Количество компаний", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, "15_company_age")
        
        results["company_age"] = {
            "mean": round(float(age_data.mean()), 2),
            "median": round(float(median_age), 2),
            "std": round(float(age_data.std()), 2),
        }
    else:
        pass
    
    return results

def generate_summary_report(
    dataset_a: pd.DataFrame,
    dataset_b: Optional[pd.DataFrame],
    missing_stats: dict[str, Any],
    target_stats: dict[str, Any],
    numerical_stats: dict[str, Any],
    categorical_stats: dict[str, Any],
    temporal_stats: dict[str, Any],
) -> None:
    """Генерирует сводный JSON-отчёт."""
    
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset_a_shape": {
                "rows": dataset_a.shape[0],
                "columns": dataset_a.shape[1],
            },
            "dataset_b_shape": {
                "rows": dataset_b.shape[0] if dataset_b is not None else 0,
                "columns": dataset_b.shape[1] if dataset_b is not None else 0,
            } if dataset_b is not None else None,
        },
        "missing_values": missing_stats,
        "target_variable": target_stats,
        "numerical_features": numerical_stats,
        "categorical_features": categorical_stats,
        "temporal_features": temporal_stats,
    }
    
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.EDA_DIR / "eda_summary.json"
    
    def make_serializable(obj):
        """Рекурсивно преобразует объекты в JSON-сериализуемый вид."""
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
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj
    
    report_serializable = make_serializable(report)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False, default=str)
    
    
def main() -> None:
    """Основной pipeline EDA."""
    logger.info("Запуск EDA")
    try:
        dataset_a, dataset_b = load_datasets()
        
        missing_stats = analyze_missing_values(dataset_a)
        
        target_stats = analyze_target_variable(dataset_a)
        
        numerical_stats = analyze_numerical_features(dataset_a)
        
        categorical_stats = analyze_categorical_features(dataset_a)
        
        temporal_stats = analyze_temporal_features(dataset_a)
        generate_summary_report(
            dataset_a, dataset_b,
            missing_stats, target_stats,
            numerical_stats, categorical_stats,
            temporal_stats
        )
        logger.info("EDA завершён; графики сохранены в %s", config.EDA_DIR)
    except Exception:
        logger.exception("Ошибка EDA")
        raise


if __name__ == "__main__":
    main()
