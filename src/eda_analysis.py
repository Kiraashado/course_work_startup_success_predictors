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
FEATURE_LABELS = {
    "team_size": "Размер команды",
    "num_founders": "Число основателей",
    "company_age": "Возраст компании",
    "success_binary": "Исход не определён",
    "crunchbase_url": "Ссылка на Crunchbase",
    "linkedin_url": "Ссылка на LinkedIn",
    "year_founded": "Год основания",
    "founders_names": "Имена основателей",
    "long_description": "Подробное описание",
    "country": "Страна",
    "city": "Город",
    "location_raw": "Местоположение",
    "short_description": "Краткое описание",
    "website": "Сайт",
    "batch_year": "Год набора",
    "top_company": "Отметка ведущей компании",
}
STATUS_LABELS = {
    "active": "Работает",
    "inactive": "Закрыта",
    "acquired": "Поглощена",
    "public": "Вышла на биржу",
    "unknown": "Неизвестно",
}
INDUSTRY_LABELS = {
    "Artificial Intelligence": "Искусственный интеллект",
    "B2B": "Решения для бизнеса",
    "Consumer": "Потребительские товары и услуги",
    "Fintech": "Финансовые технологии",
    "Healthcare": "Здравоохранение",
    "Industrials": "Промышленность",
    "Real Estate and Construction": "Недвижимость и строительство",
    "Education": "Образование",
    "Government": "Государственный сектор",
    "Unspecified": "Не указано",
}
COUNTRY_LABELS = {
    "USA": "США", "United Kingdom": "Великобритания", "India": "Индия",
    "Canada": "Канада", "Mexico": "Мексика", "France": "Франция",
    "Germany": "Германия", "Singapore": "Сингапур", "Nigeria": "Нигерия",
    "Brazil": "Бразилия", "Israel": "Израиль", "Indonesia": "Индонезия",
}
TAG_LABELS = {
    "B2B": "Решения для бизнеса",
    "SaaS": "Программное обеспечение как услуга",
    "Artificial Intelligence": "Искусственный интеллект",
    "AI": "Искусственный интеллект (AI)",
    "Fintech": "Финансовые технологии",
    "Developer Tools": "Инструменты для разработчиков",
    "Marketplace": "Торговые площадки",
    "Generative AI": "Генеративный ИИ",
    "Machine Learning": "Машинное обучение",
    "Healthcare": "Здравоохранение",
    "Consumer": "Потребительский рынок",
    "E-commerce": "Электронная торговля",
    "Analytics": "Аналитика",
    "Health Tech": "Технологии в здравоохранении",
    "Open Source": "Открытое программное обеспечение",
    "Education": "Образование",
    "Productivity": "Повышение производительности",
    "AI Assistant": "Помощники на основе ИИ",
    "Hardware": "Оборудование",
    "Payments": "Платежи",
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


def add_note(fig: Figure, text: str) -> None:
    """Добавляет к графику пояснение, необходимое для автономного чтения."""
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=8, color="#606060")


def format_percentage(value: float) -> str:
    """Форматирует долю без потери малых ненулевых значений."""
    precision = 2 if 0 < abs(value) < 0.1 else 1
    return f"{value:.{precision}f}%"


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
    if value.startswith("SPRING"):
        return "Spring"
    if value.startswith(("S", "SUMMER")):
        return "Summer"
    if value.startswith(("F", "FALL")):
        return "Fall"
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
    analyzed_df = df.drop(columns=["success_binary"], errors="ignore")
    total_cells = analyzed_df.shape[0] * analyzed_df.shape[1]
    total_missing = int(analyzed_df.isnull().sum().sum())
    missing_pct_total = total_missing / total_cells * 100 if total_cells else 0.0
    
    
    missing_per_col = analyzed_df.isnull().sum()
    missing_pct = (missing_per_col / len(analyzed_df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_pct) * 0.3)))
        bars = ax.barh(
            [FEATURE_LABELS.get(col, col) for col in missing_pct.index[::-1]],
            missing_pct.values[::-1], color=COLORS["primary"],
        )
        style_axis(
            ax,
            title="Доля незаполненных значений в данных о компаниях-выпускниках YC",
            xlabel="Доля незаполненных значений, %",
        )
        add_bar_labels(
            ax, bars, [format_percentage(value) for value in missing_pct.values[::-1]],
            horizontal=True,
        )
        ax.set_xlim(0, min(100, missing_pct.max() * 1.15))
        # add_note(
        #     fig,
        #     f"{len(df):,} компаний. Целевой признак исключён: для работающих компаний он не определён.",
        # )
        save_figure(fig, "01_missing_values")
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [STATUS_COLORS.get(s, COLORS["muted"]) for s in status_counts.index]
    status_names = [STATUS_LABELS.get(status, status) for status in status_counts.index]
    bars = ax.bar(
        status_names, status_counts.values,
        color=colors
    )
    add_bar_labels(
        ax, bars,
        [f"{value}\n({format_percentage(status_pct[status])})"
         for status, value in status_counts.items()],
    )
    style_axis(
        ax,
        title="Распределение компаний-выпускников YC по текущему состоянию",
        xlabel="Текущее состояние компании", ylabel="Число компаний",
    )
    # add_note(fig, f"{len(df):,} компаний; состояние на момент сбора данных.")
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
        [f"{value}\n({format_percentage(value / len(df_sf) * 100)})"
         for value in sf_counts.values],
    )
    style_axis(
        ax,
        title="Состав выборки компаний с известным исходом",
        xlabel="Исход", ylabel="Число компаний",
    )
    ax.set_xticks(range(len(sf_counts)), ["Успех" if s == "success" else "Неудача" for s in sf_counts.index])
    # add_note(fig, f"{len(df_sf):,} компаний. Успехом считается поглощение компании или её выход на биржу, неудачей - закрытие; работающие компании исключены.")
    add_note(fig, f"Успехом считается поглощение компании или её выход на биржу, неудачей - закрытие; работающие компании исключены.")
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
    """Анализирует числовые признаки."""
    
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
    distribution_cols = [
        col for col in ["team_size", "num_founders"] if col in valid_num_cols
    ]
    n_cols = len(distribution_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, distribution_cols):
        data = df[col].dropna()
        label = FEATURE_LABELS[col]
        
        use_log = abs(data.skew()) > 1.5 and (data >= 0).all()
        
        if use_log:
            log_data = np.log10(data + 1)
            
            ax.hist(log_data, bins=40, color=COLORS["primary"])
            
            median_val = data.median()
            ax.axvline(np.log10(median_val + 1), color=COLORS["dark"], linestyle="--",
                      linewidth=2, label=f"Медиана: {median_val:.0f}")
            
            # ax.set_title(f"{label}\nЛогарифмическая шкала", fontsize=11, fontweight="bold")
            ax.set_title(f"{label}", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{label}, человек")
            
            tick_vals = [0, 1, 10, 100, 1000, 10000]
            tick_labels = ["0", "1", "10", "100", "1K", "10K"]
            max_val = data.max()
            ax.set_xticks([np.log10(v + 1) for v in tick_vals if v <= max_val])
            ax.set_xticklabels([l for v, l in zip(tick_vals, tick_labels) 
                               if v <= max_val])
            
        else:
            cap = data.quantile(0.99)
            data_capped = data[data <= cap]
            
            lower = int(np.floor(data_capped.min()))
            upper = int(np.ceil(data_capped.max()))
            bins = np.arange(lower - 0.5, upper + 1.5)
            ax.hist(data_capped, bins=bins, color=COLORS["primary"])
            ax.set_xticks(range(lower, upper + 1))
            
            median_val = data.median()
            ax.axvline(median_val, color=COLORS["dark"], linestyle="--",
                      linewidth=2, label=f"Медиана: {median_val:.0f}")
            
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{label}, человек")
        
        ax.set_ylabel("Число компаний")
        ax.legend(fontsize=9, loc="upper right")
    
    plt.suptitle(
        "Распределение размера команды и числа основателей в компаниях-выпускниках YC",
        fontsize=14, fontweight="bold", y=1.02
    )
    add_note(
        fig,
        "Пунктиром отмечена медиана; график распределения размеров команд представлен в логарифмической шкале.",
    )
    plt.tight_layout()
    save_figure(fig, "05_numerical_distributions")

    n_box_cols = len(valid_num_cols)
    fig, axes = plt.subplots(1, n_box_cols, figsize=(5 * n_box_cols, 5))
    if n_box_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, valid_num_cols):
        data = df[col].dropna()
        label = FEATURE_LABELS[col]
        use_log = col == "team_size" and (data >= 0).all()
        plot_data = np.log10(data + 1) if use_log else data

        ax.boxplot(
            plot_data,
            orientation="vertical",
            patch_artist=True,
            boxprops={"facecolor": COLORS["primary"], "alpha": 0.85},
            medianprops={"color": COLORS["dark"], "linewidth": 2},
            whiskerprops={"color": COLORS["medium"]},
            capprops={"color": COLORS["medium"]},
            flierprops={
                "marker": "o",
                "markerfacecolor": COLORS["soft"],
                "markeredgecolor": COLORS["medium"],
                "markersize": 3,
                "alpha": 0.55,
            },
        )

        q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
        unit = " лет" if col == "company_age" else ""
        ax.text(
            0.05,
            0.95,
            f"25-й процентиль: {q1:.0f}{unit}\n"
            f"Медиана: {median:.0f}{unit}\n"
            f"75-й процентиль: {q3:.0f}{unit}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox={
                "boxstyle": "round",
                "facecolor": COLORS["light"],
                "edgecolor": COLORS["muted"],
                "alpha": 0.9,
            },
        )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylabel("Лет" if col == "company_age" else "Человек")
        ax.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax)

        if use_log:
            tick_values = [0, 1, 10, 100, 1000, 10000]
            visible_ticks = [value for value in tick_values if value <= data.max()]
            ax.set_yticks([np.log10(value + 1) for value in visible_ticks])
            ax.set_yticklabels([str(value) for value in visible_ticks])

    plt.suptitle(
        "Квартильные характеристики компаний-выпускников YC",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    add_note(
        fig,
        "Границы прямоугольника - 25-й и 75-й процентили, линия - медиана; размер команды показан в логарифмической шкале.",
    )
    plt.tight_layout()
    save_figure(fig, "06_numerical_boxplots")

    corr_matrix = None
    if len(valid_num_cols) >= 2:
        df_corr = df[valid_num_cols].dropna()
        if len(df_corr) > 0:
            corr_matrix = df_corr.corr(method="spearman")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            corr_labels = [FEATURE_LABELS[col] for col in corr_matrix.columns]
            corr_display = corr_matrix.copy()
            corr_display.index = corr_labels
            corr_display.columns = corr_labels
            sns.heatmap(
                corr_display, annot=True, fmt=".2f",
                cmap=CORRELATION_CMAP, center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, ax=ax,
                cbar=False,
            )
            ax.set_title(
                "Связь между числовыми характеристиками компаний",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=20)
            ax.tick_params(axis="y", rotation=0)
            add_note(
                fig,
                f"В клетках указана ранговая корреляция Спирмена соответствующих характеристик; вычислено на основе {len(df_corr):,} полных наблюдений.",
            )
            plt.tight_layout()
            save_figure(fig, "07_correlation_matrix")
    
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
        batch_counts = batch_counts[batch_counts["batch_year"] <= datetime.now().year]
        
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
            
            ax.set_title("Количество компаний-выпускников YC по году набора")
            ax.set_xlabel("Год набора", fontsize=11)
            ax.set_ylabel("Число компаний", fontsize=11)
            ax.set_xticks(batch_counts["batch_year"])
            ax.set_xlim(
                batch_counts["batch_year"].min() - 0.7,
                batch_counts["batch_year"].max() + 0.7,
            )
            plt.xticks(rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
            # add_note(
            #     fig,
            #     f"{int(batch_counts['count'].sum()):,} компаний с известным годом; последние наборы могут быть неполными.",
            # )
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
            season_counts = season_counts.drop(labels="Unknown", errors="ignore")
            season_counts.index = season_counts.index.map({
                "Summer": "Лето", "Winter": "Зима", "Fall": "Осень",
                "Spring": "Весна",
            })
            
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
                        f"{val}\n({format_percentage(pct)})",
                        ha="center", va="bottom", fontweight="bold"
                    )
                
                ax.set_title("Распределение компаний-выпускников YC по сезону набора")
                ax.set_ylabel("Число компаний")
                ax.set_xlabel("Сезон набора")
                ax.set_ylim(0, season_counts.max() * 1.2)
                add_note(
                    fig,
                    "Каждая компания учтена один раз; записи без указанного сезона исключены.",
                )
                
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
            [INDUSTRY_LABELS.get(industry, industry) for industry in top_industries.index[::-1]],
            top_industries.values[::-1],
            color=COLORS["primary"]
        )
        
        for bar, val in zip(bars, top_industries.values[::-1]):
            pct = val / total_with_industry * 100
            ax.text(
                val + 5, bar.get_y() + bar.get_height() / 2,
                f"{val} ({format_percentage(pct)})", va="center", fontsize=9
            )
        
        ax.set_title("Распределение компаний YC по направлениям деятельности")
        ax.set_xlabel("Число компаний", fontsize=11)
        ax.set_xlim(0, top_industries.max() * 1.15)
        add_note(
            fig,
            f"{total_with_industry:,} компаний; ИИ выделен по тегам, остальные направления - по классификации YC.",
        )
        
        plt.tight_layout()
        save_figure(fig, "11_top_industries")
        
        results["top_industries"] = {ind: int(cnt) for ind, cnt in top_industries.items()}

        yc_industries = df["industry"].dropna().astype(str).str.strip()
        yc_industries = yc_industries[yc_industries.ne("")]
        top_yc_industries = yc_industries.value_counts().head(10)
        total_yc_industries = len(yc_industries)

        if not top_yc_industries.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            bars = ax.barh(
                [
                    INDUSTRY_LABELS.get(industry, industry)
                    for industry in top_yc_industries.index[::-1]
                ],
                top_yc_industries.values[::-1],
                color=COLORS["primary"],
            )

            for bar, val in zip(bars, top_yc_industries.values[::-1]):
                pct = val / total_yc_industries * 100
                ax.text(
                    val + 5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val} ({format_percentage(pct)})",
                    va="center",
                    fontsize=9,
                )

            ax.set_title("Распределение компаний YC по исходной классификации")
            ax.set_xlabel("Число компаний", fontsize=11)
            ax.set_xlim(0, top_yc_industries.max() * 1.15)
            add_note(
                fig,
                f"{total_yc_industries:,} компаний; использовано исходное направление YC, теги не учитываются.",
            )

            plt.tight_layout()
            save_figure(fig, "11_yc_industries")

            results["top_yc_industries"] = {
                industry: int(count)
                for industry, count in top_yc_industries.items()
            }
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
                [COUNTRY_LABELS.get(country, country) for country in top_countries.index[::-1]],
                top_countries.values[::-1],
                color=COLORS["primary"]
            )
            add_bar_labels(
                ax,
                bars,
                [f"{val} ({format_percentage(val / total_with_country * 100)})"
                 for val in top_countries.values[::-1]],
                horizontal=True,
            )
            
            ax.set_title("Десять стран с наибольшим числом компаний YC")
            ax.set_xlabel("Число компаний (логарифмическая шкала)", fontsize=11)
            ax.set_xscale("log")
            add_note(
                fig,
                f"{total_with_country:,} компаний с известной страной; горизонтальная ось логарифмическая.",
            )
            
            plt.tight_layout()
            save_figure(fig, "12_top_countries")
            
            results["top_countries"] = {c: int(cnt) for c, cnt in top_countries.items()}
    else:
        pass
    if "tags" in df.columns:
        
        all_tags = []
        for tags in df["tags"]:
            normalized_tags = {
                "Artificial Intelligence" if tag == "AI" else tag
                for tag in parse_tags(tags)
            }
            all_tags.extend(normalized_tags)
        
        if all_tags:
            tag_counts = pd.Series(Counter(all_tags)).sort_values(ascending=False)
            top_tags = tag_counts.head(20)
            
            
            fig, ax = plt.subplots(figsize=(11, 8))
            bars = ax.barh(
                [TAG_LABELS.get(tag, tag) for tag in top_tags.index[::-1]],
                top_tags.values[::-1],
                color=COLORS["primary"]
            )
            
            for bar, val in zip(bars, top_tags.values[::-1]):
                ax.text(
                    val + 5, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9
                )
            
            ax.set_title("Двадцать наиболее распространённых тематических тегов компаний YC")
            ax.set_xlabel("Число компаний с тегом", fontsize=11)
            ax.set_xlim(0, top_tags.max() * 1.1)
            add_note(
                fig,
                f"Показаны 20 из {len(tag_counts):,} тегов; AI и Artificial Intelligence объединены; теги могут пересекаться.",
            )
            
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
        ).astype("Int64")
        
        df_founded = df_founded[df_founded["year_founded"].notna()]
        
        if len(df_founded) == 0:
            pass
        else:
            year_counts = df_founded["year_founded"].value_counts().sort_index()
            
            year_counts = year_counts[
                (year_counts.index >= 2005) & (year_counts.index <= 2025)
            ]
            
            if len(year_counts) > 0:
                median_year = float(df_founded["year_founded"].median())
                results["year_founded"] = {
                    "min": int(year_counts.index.min()),
                    "max": int(year_counts.index.max()),
                    "median": median_year,
                    "distribution": {int(y): int(c) for y, c in year_counts.items()},
                }
    else:
        pass
    if "company_age" in df.columns and df["company_age"].notna().sum() > 0:
        
        age_data = df["company_age"].dropna()
        
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cap = age_data.quantile(0.99)
        age_capped = age_data[age_data <= cap]
        
        lower = int(np.floor(age_capped.min()))
        upper = int(np.ceil(age_capped.max()))
        bins = np.arange(lower - 0.5, upper + 1.5)
        ax.hist(age_capped, bins=bins, color=COLORS["primary"])
        ax.set_xticks(range(lower, upper + 1, 2))
        
        median_age = age_data.median()
        ax.axvline(
            median_age, color=COLORS["dark"], linestyle="--",
            linewidth=2, label=f"Медиана: {median_age:.0f} лет"
        )
        
        ax.set_title("Распределение компаний YC по возрасту")
        ax.set_xlabel("Возраст компании (лет)", fontsize=11)
        ax.set_ylabel("Число компаний", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        add_note(
            fig,
            f"Возраст на 2025 год; {len(age_data):,} компаний; верхний 1% значений не показан.",
        )
        
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
