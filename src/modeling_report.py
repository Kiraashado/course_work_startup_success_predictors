"""Таблицы и рисунки по сохранённым проверочным прогнозам."""

import json
import hashlib
import shutil
import textwrap

import numpy as np
import pandas as pd

from config import config
from eda_analysis import BLUE, DARK, plt
from modeling import COMPARISONS, MODEL_NAMES, OUT, TEAM

FIGURES = config.PROJECT_ROOT / "coursework_text" / "graphics" / "modeling"
TABLES = config.PROJECT_ROOT / "coursework_text" / "sections" / "generated"


def decimal(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def canvas(title, protocol, size=(12, 7)):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.34, right=.95, bottom=.25, top=.79)
    fig.suptitle(title, x=.04, y=.96, ha="left", fontsize=15, fontweight="bold")
    n = f"{protocol['n_known']:,}".replace(",", " ")
    fig.text(.04, .87,
             f"YC: {n} компаний, активных 13.07.2023; статус на 01.08.2026 известен.",
             fontsize=10, color="#444444")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D5DCE3", linewidth=.6)
    return fig, ax


def save(fig, name, note):
    fig.text(.04, .03, "\n".join(textwrap.wrap(note, 125)) +
             "\nПоложительный исход: поглощение или публичный статус." +
             "\nИсточник: расчёты по сопоставленным выгрузкам каталога YC на Kaggle.",
             fontsize=9, color="#444444", va="bottom", linespacing=1.4)
    for ext in ["png", "pdf"]:
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=180)
        shutil.copyfile(path, FIGURES / path.name)
    plt.close(fig)


def metrics_plot(metrics, protocol):
    order = list(MODEL_NAMES)
    fig, ax = canvas("Ошибка вероятностного прогноза при двух схемах проверки", protocol)
    for scheme, offset, marker, color, label in [
        ("stratified", -.13, "o", BLUE, "Стратифицированная, два повтора"),
        ("grouped", .13, "s", DARK, "С разделением наборов YC"),
    ]:
        values = metrics.loc[metrics.scheme.eq(scheme)].set_index("model").loc[order, "log_loss"]
        ax.scatter(values, np.arange(len(order)) + offset, marker=marker, color=color, label=label, s=38)
    ax.set_yticks(range(len(order)), [textwrap.fill(MODEL_NAMES[m], 35) for m in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Логарифмическая потеря; меньше — лучше")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.58, .145),
               ncol=2, frameon=False, fontsize=9)
    save(fig, "01_model_quality",
         "Все прогнозы получены во внешних проверочных частях; настройка выполнена во внутренних. "
         "ЛР — логистическая регрессия; набор — год и сезон; команда — размер и число основателей. "
         "Разделение по наборам проверяет переносимость между группами, не между календарными периодами.")


def comparisons_plot(comparisons, protocol):
    table = comparisons.loc[comparisons.scheme.eq("stratified")].set_index("comparison")
    table = table.loc[[row[2] for row in COMPARISONS]]
    fig, ax = canvas("Изменение ошибки при добавлении признаков и смене модели", protocol, (12, 8))
    fig.subplots_adjust(left=.43, bottom=.25)
    values = table.gain_log_loss.to_numpy()
    for i, (_, row) in enumerate(table.iterrows()):
        ax.plot([row.conditional_low, row.conditional_high], [i, i], color=BLUE, linewidth=2)
    ax.scatter(values, range(len(table)), color=DARK, s=40)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(table)), [textwrap.fill(s, 36) for s in table.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Уменьшение логарифмической потери; справа от нуля — улучшение")
    save(fig, "02_incremental_value",
         "ЛР — логистическая регрессия; команда — размер и число основателей. Парные разности потерь усреднены по двум повторам. "
         "Отрезки — центральные 95% из 2000 "
         "повторных выборок компаний при фиксированных прогнозах. Переобучение и зависимость внутри наборов не учтены; это не полная неопределённость процедуры.")


def calibration_plot(predictions, protocol):
    pred = predictions.loc[predictions.scheme.eq("stratified") & predictions.repeat.eq(0)]
    fig, ax = canvas("Предсказанные вероятности и наблюдаемые частоты исхода", protocol, (11, 7))
    fig.subplots_adjust(left=.12, right=.95, bottom=.28)
    rows = []
    for model, color, marker in [("lr_team", BLUE, "o"), ("cb_age", "#A7C2DA", "s"), ("cb_team", DARK, "D")]:
        group = pred.loc[pred.model.eq(model)].copy()
        group["bin"] = pd.qcut(group.probability, q=5, duplicates="drop")
        table = group.groupby("bin", observed=True).agg(predicted=("probability", "mean"),
                  observed=("y", "mean"), n=("y", "size"), positives=("y", "sum"))
        ax.plot(table.predicted, table.observed, marker=marker, color=color, label=MODEL_NAMES[model])
        rows.extend({"model": model, "bin": str(k), **v} for k, v in table.to_dict("index").items())
    limit = max(max(row["predicted"], row["observed"]) for row in rows) * 1.2
    ax.plot([0, limit], [0, limit], color="#888888", linestyle="--", label="Совпадение вероятности и частоты")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Средняя предсказанная вероятность", ylabel="Доля поглощённых и публичных компаний")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .15), ncol=2, frameon=False, fontsize=9)
    pd.DataFrame(rows).to_csv(OUT / "calibration_bins.csv", index=False)
    save(fig, "03_calibration",
         "Первый стратифицированный повтор; только проверочные прогнозы. Точки — группы примерно равного размера по "
         "вероятности, границы определены отдельно для каждой модели. Линии соединяют точки; дополнительная калибровка моделей не выполнялась.")


def shap_plot(shap, protocol):
    names = {"batch_year": "Год набора", "batch_season": "Сезон набора", "company_age": "Возраст компании",
             "log_team_size": "Логарифм размера команды: ln(1 + размер)", "num_founders": "Число основателей"}
    importance = shap[TEAM].abs().mean().sort_values(ascending=False)
    importance.rename("mean_absolute_shap").to_csv(OUT / "shap_importance.csv", index_label="feature")
    fig, ax = canvas("Вклад признаков в проверочные прогнозы CatBoost", protocol)
    ax.barh(range(len(importance)), importance, color=BLUE, height=.6)
    ax.set_yticks(range(len(importance)), [textwrap.fill(names[c], 28) for c in importance.index])
    ax.invert_yaxis()
    ax.set_xlim(0, importance.max()*1.25)
    for i, value in enumerate(importance):
        ax.text(value+importance.max()*.02, i, decimal(value, 3), va="center", fontsize=10)
    ax.set_xlabel("Средний модуль SHAP; единицы логарифма шансов")
    save(fig, "04_shap_importance",
         "CatBoost с набором, возрастом и командой; первый стратифицированный повтор. Для каждой компании объясняется "
         "модель, не обучавшаяся на ней. Модуль SHAP не показывает направление связи, причинный эффект или улучшение проверочного качества.")


def export_tables(metrics, comparisons):
    for scheme in ["stratified", "grouped"]:
        table = metrics.loc[metrics.scheme.eq(scheme)].set_index("model").loc[list(MODEL_NAMES)]
        rows = [MODEL_NAMES[m] + " & " + " & ".join(decimal(row[c]) for c in
                ["log_loss", "average_precision", "roc_auc", "brier"]) + r" \\" for m, row in table.iterrows()]
        header = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{7.1cm}rrrr@{}}" + "\n" +
                  r"\toprule" + "\n" + r"Модель & LogLoss & AP & ROC-AUC & Брайер \\" + "\n" + r"\midrule" + "\n")
        (TABLES / f"metrics_{scheme}.tex").write_text(header + "\n".join(rows) + "\n" +
                                                   r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")
    table = comparisons.loc[comparisons.scheme.eq("stratified")]
    rows = [row.comparison + " & " + decimal(row.gain_log_loss, 5) + " & [" +
            decimal(row.conditional_low, 5) + "; " + decimal(row.conditional_high, 5) + r"] \\"
            for row in table.itertuples()]
    header = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{8.1cm}rr@{}}" + "\n" +
              r"\toprule" + "\n" + r"Сравнение & $\Delta L$ & Условные границы \\" + "\n" + r"\midrule" + "\n")
    (TABLES / "model_comparisons.tex").write_text(header + "\n".join(rows) + "\n" +
                                                r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if not (OUT / "results.json").is_file():
        raise RuntimeError("Сначала требуется завершить modeling.py")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    if result["completed_at"] < protocol["created_at"]:
        raise RuntimeError("Последний запуск обучения ещё не завершён")
    cohort_path = config.PREPROCESSED_DIR / "analysis_cohort.parquet"
    if hashlib.sha256(cohort_path.read_bytes()).hexdigest() != protocol["cohort_sha256"]:
        raise RuntimeError("Когорта изменилась после обучения моделей")
    metrics = pd.read_csv(OUT / "metrics.csv")
    comparisons = pd.read_csv(OUT / "comparisons.csv")
    predictions = pd.read_csv(OUT / "predictions.csv")
    shap = pd.read_csv(OUT / "shap_oof.csv")
    metrics_plot(metrics, protocol)
    comparisons_plot(comparisons, protocol)
    calibration_plot(predictions, protocol)
    shap_plot(shap, protocol)
    export_tables(metrics, comparisons)
    print(f"Подготовлены 4 рисунка и таблицы: {FIGURES}")


if __name__ == "__main__":
    main()
