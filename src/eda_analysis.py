"""Разведочный анализ когорты YC: признаки 13.07.2023, статусы 01.08.2026."""

import json
import os
import shutil
import textwrap
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "artifacts" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from config import config
from data_preprocessing import BASELINE_FEATURES, save_json
from utils.logger import setup_logger

logger = setup_logger("eda", log_file=config.ARTIFACTS_DIR / "eda.log")
BLUE = "#37679B"
DARK = "#1E4268"
PALETTE = {
    "active": BLUE, "inactive": DARK, "acquired": "#7298BE",
    "public": "#A7C2DA", "unknown": "#D4E1ED",
}
STATUS_ORDER = ["active", "inactive", "acquired", "public", "unknown"]
STATUS_NAMES = {
    "active": "Активна", "inactive": "Неактивна", "acquired": "Поглощена",
    "public": "Публичная", "unknown": "Статус неизвестен",
}
COUNTRIES = {
    "US": "США", "IN": "Индия", "CA": "Канада", "GB": "Великобритания",
    "MX": "Мексика", "NG": "Нигерия", "BR": "Бразилия", "SG": "Сингапур",
    "ID": "Индонезия", "DE": "Германия", "FR": "Франция", "CO": "Колумбия",
}
TAGS = {
    "SaaS": "Программное обеспечение как услуга",
    "B2B": "Решения для бизнеса", "Fintech": "Финансовые технологии",
    "Developer Tools": "Инструменты для разработчиков",
    "Artificial Intelligence": "Искусственный интеллект (Artificial Intelligence)",
    "Marketplace": "Торговые площадки", "Machine Learning": "Машинное обучение",
    "E-commerce": "Электронная торговля", "Climate": "Климатические технологии",
    "AI": "Искусственный интеллект (AI)", "Education": "Образование",
    "Consumer": "Потребительские товары и услуги",
    "Generative AI": "Генеративный ИИ", "Analytics": "Аналитика",
    "Open Source": "Открытое программное обеспечение",
}
NUMERIC_LABELS = {
    "team_size": "Размер команды", "num_founders": "Число основателей",
    "company_age": "Возраст компании", "tag_count": "Число тегов",
    "description_length": "Длина краткого описания",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#AAAAAA",
    "axes.unicode_minus": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    "pdf.fonttype": 42,
})
MANIFEST = []


def number(x):
    return f"{x:,.0f}".replace(",", " ")


def percent(x):
    return f"{x:.1f}".replace(".", ",") + "%"


def figure(title, n, *, cols=1, size=(11, 6.5)):
    fig, axs = plt.subplots(1, cols, figsize=size, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.23, top=0.80, wspace=0.33)
    fig.suptitle(title, x=0.04, y=0.97, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.04, 0.885,
             f"YC: {number(n)} компаний, активных 13.07.2023. Последующие статусы: 01.08.2026.",
             fontsize=10, color="#444444")
    return fig, axs[0]


def save_figure(fig, name, note):
    source = "Источник: выгрузки каталога YC на Kaggle; расчёты по исходному срезу и сопоставление по id."
    text = "\n".join(textwrap.wrap(note, 132)) + "\n" + source
    fig.text(0.04, 0.03, text, va="bottom", fontsize=8.5, color="#444444", linespacing=1.4)
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    config.COHORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        path = config.EDA_DIR / f"{name}.{suffix}"
        fig.savefig(path, dpi=180)
        shutil.copyfile(path, config.COHORT_FIGURES_DIR / path.name)
    MANIFEST.append({"file": name, "note": note})
    plt.close(fig)


def grid(ax, direction="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=direction, color="#D5DCE3", linewidth=0.6)


def hbars(ax, values, labels, denominator, *, xmax=None):
    bars = ax.barh(range(len(values)), values, color=BLUE, height=0.65)
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax or max(max(values) * 1.32, 1))
    for bar, value in zip(bars, values):
        ax.text(value + ax.get_xlim()[1] * 0.012, bar.get_y() + bar.get_height()/2,
                f"{number(value)} ({percent(100*value/denominator)})", va="center", fontsize=9)
    grid(ax)


def describe(values):
    missing = int(values.isna().sum())
    values = values.dropna().astype(float)
    if values.empty:
        return {"n": 0, "missing": missing}
    return {
        "n": len(values), "missing": missing,
        "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
        "q1": float(values.quantile(.25)), "median": float(values.median()),
        "q3": float(values.quantile(.75)), "p90": float(values.quantile(.9)),
        "p99": float(values.quantile(.99)),
    }


def missing_plot(df):
    missing = {NUMERIC_LABELS.get(col, col): int(df[col].isna().sum())
               for col in ["team_size","num_founders","company_age","description_length"]}
    missing.update({"Страна": int(df.country.isna().sum()),
                    "Подробное описание": int(df.long_description.isna().sum()),
                    "Нет перечисленных тегов": int(df.tag_count.eq(0).sum())})
    missing = dict(sorted(missing.items(), key=lambda item: -item[1]))
    fig, (ax,) = figure("Полнота исходных характеристик компаний", len(df))
    fig.subplots_adjust(left=.30)
    hbars(ax, list(missing.values()), list(missing), len(df))
    ax.set_xlabel("Число компаний без указанных сведений")
    save_figure(fig, "02_missing_values",
                "Сведения на 13.07.2023. Пустой список тегов означает отсутствие перечисленных тегов. "
                "Нулевое число основателей признано неизвестным; поздними значениями пропуски не заполнены.")
    return missing


def status_plot(df):
    counts = df.status_followup.value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig, (ax,) = figure("Статусы компаний к концу периода наблюдения", len(df))
    fig.subplots_adjust(left=.25)
    hbars(ax, counts.tolist(), [STATUS_NAMES[s] for s in counts.index], len(df))
    ax.set_xlabel("Число компаний; в скобках доля исходной когорты")
    save_figure(fig, "03_followup_status",
                "Показатель успеха: поглощение или публичный статус. Активные компании не исключены. "
                "Неизвестный статус не приравнивается к неактивности; статус каталога не задаёт точную дату события.")
    return {key: int(value) for key, value in counts.items()}


def cohort_group(row):
    if not row.regular_batch:
        return str(row.batch) if pd.notna(row.batch) else "Не определён"
    year = row.batch_year
    if pd.isna(year):
        return "Не определён"
    if year <= 2014:
        return "2006-2014"
    if year <= 2017:
        return "2015-2017"
    if year <= 2020:
        return "2018-2020"
    return "2021-2023"


def cohort_plot(df):
    groups = df.apply(cohort_group, axis=1)
    table = pd.crosstab(groups, df.status_followup).reindex(columns=STATUS_ORDER, fill_value=0)
    order = [g for g in ["2006-2014","2015-2017","2018-2020","2021-2023","IK12"] if g in table.index]
    order += sorted(set(table.index) - set(order))
    table = table.reindex(order)
    table.to_csv(config.EDA_DIR / "cohort_status_counts.csv", index_label="batch_group")
    fig, axs = figure("Годы набора и последующие статусы исходной когорты", len(df), cols=2, size=(13, 7))
    fig.subplots_adjust(left=.12, right=.97, bottom=.30, wspace=.22)
    totals = table.sum(axis=1)
    axs[0].barh(order, totals, color=BLUE)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, totals.max()*1.2)
    for y, n in enumerate(totals):
        axs[0].text(n+15,y,number(n),va="center",fontsize=10)
    axs[0].set_xlabel("Число компаний")
    axs[0].set_ylabel("Год набора или обозначение в источнике")
    grid(axs[0])
    left = np.zeros(len(table))
    for s in STATUS_ORDER:
        values = table[s].to_numpy()/totals.to_numpy()*100
        axs[1].barh(order, values, left=left, label=STATUS_NAMES[s], color=PALETTE[s])
        for y, p in enumerate(values):
            if p >= 8:
                axs[1].text(left[y]+p/2,y,percent(p),ha="center",va="center",
                            color="white" if s in {"active","inactive"} else DARK,fontsize=9)
        left += values
    axs[1].invert_yaxis()
    axs[1].set_xlim(0,100)
    axs[1].set_xlabel("Доля компаний внутри группы, %")
    axs[1].set_yticklabels([])
    fig.legend(*axs[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5,.16),
               ncol=5, frameon=False, fontsize=9)
    save_figure(fig,"04_cohort_composition",
                "Группы заданы по году набора; малочисленные ранние годы объединены. "
                "IK12 показан отдельно от стандартных сезонных наборов. На правой панели знаменатель — все компании группы, включая неизвестные статусы.")
    return {str(k): {s: int(v) for s,v in row.items()} for k,row in table.to_dict("index").items()}


def numerical_plot(df):
    fig, axs = figure("Размер команды, число основателей и возраст компаний", len(df), cols=3, size=(14,6.5))
    fig.subplots_adjust(left=.06, right=.97, wspace=.30)
    team = df.team_size.dropna().astype(float)
    edges = np.array([0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000])
    axs[0].hist(team, bins=edges, color=BLUE, edgecolor="white")
    axs[0].set_xlim(0, edges[-1])
    axs[0].set_xscale("function", functions=(np.log1p, np.expm1))
    axs[0].set_xticks([0,1,5,20,100,1000,10000], ["0","1","5","20","100","1 000","10 000"])
    axs[0].tick_params(axis="x",labelsize=8)
    axs[0].set_xlabel("Человек; шкала ln(1 + размер)")
    axs[0].set_title(f"Размер команды, n = {number(len(team))}")
    founders = df.num_founders.dropna().astype(int).value_counts().sort_index()
    axs[1].bar(founders.index,founders.values,color=BLUE)
    axs[1].set_xticks(founders.index)
    axs[1].set_xlabel("Основателей, человек")
    axs[1].set_title(f"Число основателей, n = {number(founders.sum())}")
    age = df.company_age.dropna().astype(float)
    axs[2].hist(age,bins=np.arange(-.5,age.max()+1.5),color=BLUE,edgecolor="white")
    axs[2].set_xlabel("2023 минус год основания, лет")
    axs[2].set_title(f"Приближённый возраст, n = {number(len(age))}")
    for ax in axs:
        ax.set_ylabel("Число компаний")
        grid(ax,"y")
    save_figure(fig, "05_numerical_distributions",
                "Все признаки измерены на 13.07.2023, пропуски исключены отдельно для каждой панели. "
                "Размер команды: высота столбца — число компаний в интервале, не плотность; показаны все значения, включая нули.")


def boxplot(df):
    labels, groups = [], []
    stats = {}
    for s in STATUS_ORDER:
        raw_values = df.loc[df.status_followup.eq(s),"team_size"]
        values = raw_values.dropna().astype(float)
        stats[s] = describe(raw_values)
        if len(values):
            stats[s]["share_le_10"] = float(values.le(10).mean())
            stats[s]["share_gt_50"] = float(values.gt(50).mean())
            labels.append(f"{STATUS_NAMES[s]}\nn = {number(len(values))}")
            groups.append((s, values.to_numpy()))
    fig, (ax,) = figure("Исходный размер команды и последующий статус компании", len(df), size=(12,7))
    fig.subplots_adjust(left=.23, right=.96, bottom=.27)
    for position, (_, values) in enumerate(groups, start=1):
        transformed = np.log1p(values)
        if len(values) >= 20:
            cutoff = np.quantile(values, .99)
            core = values[values <= cutoff]
            violin = ax.violinplot(np.log1p(core), positions=[position], orientation="horizontal",
                                   widths=.75, showextrema=False, points=300)
            for body in violin["bodies"]:
                body.set(facecolor="#A7C2DA", edgecolor=BLUE, linewidth=1, alpha=.75)
            tail = np.log1p(np.sort(values[values > cutoff]))
            if len(tail):
                ax.scatter(tail, np.full(tail.shape, position),
                           s=14, color=BLUE, edgecolor="white", linewidth=.3, zorder=3)
        else:
            ax.scatter(transformed, np.full(transformed.shape, position), s=19, color=BLUE,
                       edgecolor="white", linewidth=.4, zorder=3)
        box = boxplot_stats(values, whis=1.5)[0]
        for key in ("med", "q1", "q3", "whislo", "whishi"):
            box[key] = np.log1p(box[key])
        ax.bxp([box], positions=[position], orientation="horizontal", widths=.17,
               showfliers=False, patch_artist=True,
               boxprops={"facecolor":"white", "edgecolor":DARK, "linewidth":1.1},
               whiskerprops={"color":DARK, "linewidth":1.1},
               capprops={"color":DARK, "linewidth":1.1},
               medianprops={"color":DARK, "linewidth":2})
    ax.set_yticks(range(1,len(labels)+1),labels)
    ax.invert_yaxis()
    ax.set_xlim(0, np.log1p(max(max(v) for _, v in groups)) * 1.03)
    ticks=[0,1,5,10,20,50,100,500,1000,5000]
    ax.set_xticks(np.log1p(ticks), [number(v) for v in ticks])
    ax.set_xlabel("Размер команды на 13.07.2023, человек; шкала ln(1 + размер)")
    grid(ax)
    save_figure(fig,"06_team_size_boxplot",
                "Группы заданы статусом на 01.08.2026. Скрипки: плотность на логарифмической шкале без значений "
                "выше 99-го процентиля каждой группы; ширина нормирована отдельно. Эти значения показаны точками. "
                "Коробки и усы рассчитаны по всем данным до преобразования шкалы. Семь публичных компаний показаны точками без оценки плотности.")
    return stats


def correlation_plot(df):
    cols = list(NUMERIC_LABELS)
    numeric = df[cols].astype(float)
    corr = numeric.corr(method="spearman", min_periods=3)
    present = numeric.notna().astype(int)
    pair_n = present.T @ present
    corr.to_csv(config.EDA_DIR / "spearman.csv", index_label="feature")
    pair_n.to_csv(config.EDA_DIR / "spearman_pair_counts.csv", index_label="feature")
    fig, (ax,) = figure("Ранговые связи между исходными числовыми признаками", len(df), size=(10.5,8))
    fig.subplots_adjust(left=.27, right=.91, bottom=.27)
    cmap = LinearSegmentedColormap.from_list("blue_magnitude",["#F5F8FB",BLUE])
    ax.imshow(np.abs(corr.to_numpy()), vmin=0,vmax=1,cmap=cmap,aspect="auto")
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i,j]
            label = f"{value:+.2f}".replace(".",",") if pd.notna(value) else "не определена"
            ax.text(j,i,f"{label}\nn={pair_n.iloc[i,j]}",ha="center",va="center",fontsize=9,
                    color="white" if pd.notna(value) and abs(value)>.65 else DARK)
    labels=[textwrap.fill(NUMERIC_LABELS[c],12) for c in cols]
    ax.set_xticks(range(len(cols)),labels,fontsize=9)
    ax.set_yticks(range(len(cols)),labels,fontsize=9)
    save_figure(fig,"07_correlation_matrix",
                "Коэффициент Спирмена, расчёт по попарно полным наблюдениям; n — число пар. "
                "Насыщенность синего показывает модуль, знак указан числом. Возраст = 2023 − год основания; длина описания — число символов. Целевая переменная не включена.")
    return {
        c: {other: float(value) if pd.notna(value) else None for other,value in corr[c].items()} for c in cols
    }


def countries_plot(df):
    counts = df.country.dropna().value_counts()
    top = counts.head(10)
    fig,(ax,) = figure("Десять наиболее представленных стран исходной когорты",len(df))
    fig.subplots_adjust(left=.21,right=.78)
    ax.scatter(top.values,range(len(top)),s=50,color=BLUE)
    ax.set_yticks(range(len(top)),[COUNTRIES.get(c,c) for c in top.index])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(max(.8,top.min()*.65),top.max()*4)
    ax.set_xticks([20,50,100,500,2000],[number(x) for x in [20,50,100,500,2000]])
    for y,n in enumerate(top):
        ax.annotate(f"{number(n)} ({percent(n/len(df)*100)})",(n,y),xytext=(7,0),
                    textcoords="offset points",va="center",fontsize=9)
    ax.set_xlabel("Число компаний; логарифмическая шкала")
    grid(ax)
    save_figure(fig,"08_top_countries",
                f"Страна из исходного среза, без восстановления по поздним сведениям. Проценты от всех {number(len(df))} компаний. "
                f"Страна не указана у {number(df.country.isna().sum())}; остальные страны вне первой десятки не показаны.")
    return {k:int(v) for k,v in counts.items()}


def tags_plot(df):
    frequencies = Counter(tag for tags in df.tags for tag in set(tags))
    counts = pd.Series(dict(sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))), dtype=int)
    top = counts.head(15)
    fig,(ax,) = figure("Пятнадцать наиболее частых тегов деятельности компаний",len(df),size=(12,9))
    fig.subplots_adjust(left=.38,top=.81,bottom=.18,right=.88)
    labels=[textwrap.fill(TAGS.get(t,t),32) for t in top.index]
    hbars(ax,top.tolist(),labels,len(df))
    ax.set_xlabel("Число компаний с тегом; доля всей когорты")
    ax.tick_params(axis="y",labelsize=9)
    save_figure(fig,"09_top_tags",
                f"Теги на 13.07.2023: одна компания может иметь несколько тегов; сумма долей не обязана равняться 100%. "
                f"Пустые списки: {number(df.tag_count.eq(0).sum())}. AI и Artificial Intelligence сохранены как разные исходные теги; это не классификация отраслей.")
    return {k:int(v) for k,v in counts.items()}


def cohort_diagnostics(df):
    complete_fields = ["team_size", "num_founders", "company_age", "country", "description_length"]
    complete = df[complete_fields].notna().all(axis=1)
    groups = {}
    for name, mask in {"known": df.outcome_known, "unknown": ~df.outcome_known}.items():
        subset = df.loc[mask]
        groups[name] = {
            "n": len(subset),
            "missing_age": int(subset.company_age.isna().sum()),
            "complete_cases": int(complete.loc[mask].sum()),
        }
    return {
        "complete_case_fields": complete_fields,
        "complete_cases": int(complete.sum()),
        "by_outcome_availability": groups,
        "founder_counts": {str(k): int(v) for k, v in df.num_founders.dropna().astype(int).value_counts().sort_index().items()},
        "zero_team_size": int(df.team_size.eq(0).sum()),
        "batch_season_counts": df.batch_season.fillna("unknown").value_counts().to_dict(),
    }


def main():
    logger.info("Запуск EDA исходной когорты")
    config.EDA_DIR.mkdir(parents=True,exist_ok=True)
    MANIFEST.clear()
    report = json.loads((config.PREPROCESSED_DIR/"cohort_report.json").read_text(encoding="utf-8"))
    if report["baseline_date"] != config.BASELINE_DATE or report["followup_date"] != config.FOLLOWUP_DATE:
        raise ValueError("Даты подготовленных данных не соответствуют настройкам")
    df = pd.read_parquet(config.PREPROCESSED_DIR/"analysis_cohort.parquet")
    if not df.id.is_unique or not df.status_baseline.eq("active").all() or len(df) != report["cohort"]["baseline_active"]:
        raise ValueError("Нарушена структура исходной когорты")
    if not df.loc[~df.outcome_known,"success_binary"].isna().all():
        raise ValueError("Неизвестные статусы получили целевую метку")
    missing = missing_plot(df)
    statuses = status_plot(df)
    cohorts = cohort_plot(df)
    numerical_plot(df)
    boxes = boxplot(df)
    correlations = correlation_plot(df)
    countries = countries_plot(df)
    tags = tags_plot(df)
    known = int(df.outcome_known.sum())
    positives = int(df.success_binary.eq(1).sum())
    unknown = len(df)-known
    summary = {
        "dates": {"baseline": config.BASELINE_DATE,"followup":config.FOLLOWUP_DATE},
        "n":len(df),"known_outcomes":known,"unknown_outcomes":unknown,"positive_outcomes":positives,
        "positive_fraction_known": positives/known if known else None,
        "positive_fraction_bounds_all": [positives/len(df),(positives+unknown)/len(df)],
        "missing":missing,"statuses":statuses,"cohorts":cohorts,
        "numerical":{col:describe(df[col]) for col in NUMERIC_LABELS},
        "team_by_status":boxes,"correlations":correlations,"countries":countries,"tags":tags,
        "diagnostics": cohort_diagnostics(df),
        "source_sha256": {key:value["sha256"] for key,value in report["sources"].items()},
        "figures":MANIFEST,
    }
    save_json(summary,config.EDA_DIR/"eda_summary.json")
    df[BASELINE_FEATURES+["id","status_followup","outcome_known","success_binary"]].to_parquet(
        config.EDA_DIR/"eda_data.parquet", index=False)
    logger.info("EDA: %d компаний, %d рисунков; %s",len(df),len(MANIFEST),config.EDA_DIR)


if __name__ == "__main__":
    main()
