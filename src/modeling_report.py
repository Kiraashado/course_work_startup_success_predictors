"""Таблицы и рисунки по сохранённым проверочным прогнозам."""

import json
import hashlib
import shutil
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from config import config
from eda_analysis import BLUE, DARK, plt
from modeling import COMPARISONS, MODEL_NAMES, OUT, TEAM

FIGURES = config.PROJECT_ROOT / "coursework_text" / "graphics" / "modeling"
TABLES = config.PROJECT_ROOT / "coursework_text" / "sections" / "generated"
PROBABILITY_ORDER = ["constant", "lr_base", "lr_team", "lr_geo", "lr_tags", "lr_text", "lr_all", "cb_team"]
RANKING_ORDER = ["lr_team", "lr_geo", "lr_tags", "lr_text", "lr_all", "lr_all_ap", "lr_all_balanced", "cb_team"]


def decimal(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def canvas(title, protocol, size=(12, 7)):
    size = (min(size[0], 10), size[1])
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.34, right=.95, bottom=.25, top=.79)
    fig.suptitle(textwrap.fill(title, 75), x=.04, y=.96, ha="left", fontsize=14, fontweight="bold")
    n = f"{protocol['n_known']:,}".replace(",", " ")
    fig.text(.04, .87,
             f"YC: {n} компаний, активных 13.07.2023; статус известен в августовской версии 2026 года.",
             fontsize=10, color="#444444")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D5DCE3", linewidth=.6)
    return fig, ax


def save(fig, name, note):
    for ax in fig.axes:
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=9 if len(ax.get_yticks()) > 25 else 11)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
    fig.text(.04, .03, "\n".join(textwrap.wrap(note, 108)) +
             "\nИсточник: расчёты автора по выгрузкам YC с Kaggle.",
             fontsize=10, color="#444444", va="bottom", linespacing=1.3)
    for ext in ["png", "pdf"]:
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=180)
        shutil.copyfile(path, FIGURES / path.name)
    plt.close(fig)


def metrics_plot(metrics, protocol):
    order = PROBABILITY_ORDER
    fig, ax = canvas("Ошибка вероятностного прогноза при двух схемах проверки", protocol)
    for scheme, offset, marker, color, label in [
        ("stratified", -.13, "o", BLUE, "Случайное разделение, пять повторов"),
        ("grouped", .13, "s", DARK, "Разделение наборов YC, пять повторов"),
    ]:
        values = metrics.loc[metrics.scheme.eq(scheme)].set_index("model").loc[order, "log_loss"]
        ax.scatter(values, np.arange(len(order)) + offset, marker=marker, color=color, label=label, s=38)
    ax.set_yticks(range(len(order)), [textwrap.fill(MODEL_NAMES[m], 35) for m in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Логарифмическая потеря; меньше — лучше")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.58, .145),
               ncol=2, frameon=False, fontsize=9)
    save(fig, "01_model_quality",
         "Все прогнозы получены во внешних проверочных частях; словари тегов и текста также оценивались только на обучающих данных. "
         "Разделение по наборам проверяет переносимость между группами, не между календарными периодами.")


def comparisons_plot(comparisons, protocol):
    table = comparisons.loc[comparisons.scheme.eq("grouped")].set_index("comparison")
    table = table.loc[[row[2] for row in COMPARISONS]]
    fig, ax = canvas("Изменение ошибки при добавлении признаков и смене модели", protocol, (12, 10))
    fig.subplots_adjust(left=.43, bottom=.25)
    values = table.gain_log_loss.to_numpy()
    for i, (_, row) in enumerate(table.iterrows()):
        ax.plot([row.cluster_low, row.cluster_high], [i, i], color=BLUE, linewidth=2)
    ax.scatter(values, range(len(table)), color=DARK, s=40)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(table)), [textwrap.fill(s, 36) for s in table.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Уменьшение логарифмической потери; справа от нуля — улучшение")
    save(fig, "02_incremental_value",
         "Парные разности потерь усреднены по пяти повторам с разделением наборов YC. "
         "Отрезки — центральные 95% из 2000 "
         "повторных выборок целых наборов при фиксированных прогнозах; переобучение не учтено.")


def calibration_plot(predictions, protocol):
    pred = predictions.loc[predictions.scheme.eq("grouped") & predictions.repeat.eq(0)]
    fig, ax = canvas("Предсказанные вероятности и наблюдаемые частоты исхода", protocol, (11, 7))
    fig.subplots_adjust(left=.12, right=.95, bottom=.32)
    rows = []
    for model, color, marker in [("lr_team", "#A7C2DA", "o"), ("lr_all", BLUE, "s"), ("cb_team", DARK, "D")]:
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
         "Первый повтор с разделением наборов YC; только проверочные прогнозы. Точки — группы примерно равного размера по "
         "вероятности, границы определены отдельно для каждой модели. Линии соединяют точки; дополнительная калибровка моделей не выполнялась.")


def precision_recall_plot(predictions, metrics, protocol):
    grouped = predictions.loc[predictions.scheme.eq("grouped")]
    prevalence = float(grouped.y.mean())
    styles = {
        "lr_team": ("#A7C2DA", "--"),
        "lr_tags": ("#6F9FC5", "-."),
        "lr_all": (BLUE, "-"),
        "lr_all_ap": ("#BC6C25", "-."),
        "lr_all_balanced": (DARK, ":"),
    }
    fig, ax = canvas("Точность и полнота при редком положительном исходе", protocol, (11, 7))
    fig.subplots_adjust(left=.12, right=.95, bottom=.32)
    recall_grid = np.linspace(0, 1, 301)
    grouped_metrics = metrics.loc[metrics.scheme.eq("grouped")].set_index("model")
    for model, (color, linestyle) in styles.items():
        mean_precision = mean_pr_curve(grouped.loc[grouped.model.eq(model)], recall_grid)
        ap = grouped_metrics.loc[model, "average_precision"]
        ax.plot(recall_grid, mean_precision, color=color, linestyle=linestyle, linewidth=2,
                label=f"{MODEL_NAMES[model]}: AP={decimal(ap, 3)}")
    ax.axhline(prevalence, color="#888888", linewidth=1, linestyle="--",
               label=f"Доля положительного класса: {decimal(prevalence * 100, 1)}%")
    ax.set(xlim=(0, 1), ylim=(0, max(.3, ax.get_ylim()[1])),
           xlabel="Полнота: доля найденных положительных исходов",
           ylabel="Точность: доля положительных среди отобранных")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .16),
               ncol=2, frameon=False, fontsize=9)
    save(fig, "04_precision_recall",
         "Кривые отдельных частей перенесены на общую сетку полноты ступенчатой интерполяцией и усреднены с весами по размеру частей. "
         "AP усреднена отдельно; это не площадь под нарисованной кривой. Обе модели для проверки весов подбираются по AP.")


def top_decile_plot(metrics, protocol):
    table = metrics.loc[metrics.scheme.eq("grouped")].set_index("model").loc[RANKING_ORDER]
    baseline = float(table.prevalence.iloc[0])
    fig, ax = canvas("Доля положительных исходов среди 10% наибольших оценок", protocol, (12, 7))
    values = table.precision_at_10.to_numpy()
    ax.barh(range(len(table)), values, color=BLUE, height=.58)
    ax.axvline(baseline, color="#888888", linestyle="--", linewidth=1,
               label=f"Доля во всей выборке: {decimal(baseline * 100, 1)}%")
    ax.set_yticks(range(len(table)), [textwrap.fill(MODEL_NAMES[m], 34) for m in table.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Доля поглощённых или публичных компаний")
    for i, value in enumerate(values):
        ax.text(value + .003, i, f"{decimal(value * 100, 1)}%", va="center", fontsize=9)
    ax.set_xlim(0, values.max()*1.27)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "05_top_decile",
         "В каждой части отобраны верхние 10%. Точность рассчитана по сумме отобранных и найденных компаний внутри повтора; затем усреднена по пяти повторам. "
         "Метрика оценивает качество ограниченного по объёму отбора, а не калибровку вероятностей.")


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
    save(fig, "06_shap_importance",
         "CatBoost с набором, возрастом и командой; первый групповой повтор. Для каждой компании объясняется "
         "модель, не обучавшаяся на ней. Модуль SHAP не показывает направление связи, причинный эффект или улучшение проверочного качества.")


def export_tables(metrics, comparisons):
    for scheme in ["stratified", "grouped"]:
        table = metrics.loc[metrics.scheme.eq(scheme)].set_index("model").loc[PROBABILITY_ORDER]
        rows = [MODEL_NAMES[m] + " & " + " & ".join(decimal(row[c]) for c in
                ["log_loss", "average_precision", "roc_auc", "brier"]) + r" \\" for m, row in table.iterrows()]
        header = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{7.1cm}rrrr@{}}" + "\n" +
                  r"\toprule" + "\n" + r"Модель & LogLoss & AP & ROC-AUC & Брайер \\" + "\n" + r"\midrule" + "\n")
        (TABLES / f"metrics_{scheme}.tex").write_text(header + "\n".join(rows) + "\n" +
                                                   r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")
    table = metrics.loc[metrics.scheme.eq("grouped")].set_index("model").loc[RANKING_ORDER]
    rows = [MODEL_NAMES[m] + " & " + " & ".join(decimal(row[c]) for c in
            ["average_precision", "roc_auc", "precision_at_10", "recall_at_10", "lift_at_10"]) + r" \\"
            for m, row in table.iterrows()]
    header = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{6.4cm}rrrrr@{}}" + "\n" +
              r"\toprule" + "\n" + r"Модель & AP & ROC-AUC & P@10\% & R@10\% & Lift@10\% \\" + "\n" + r"\midrule" + "\n")
    (TABLES / "ranking_grouped.tex").write_text(header + "\n".join(rows) + "\n" +
                                                r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")
    table = comparisons.loc[comparisons.scheme.eq("grouped")]
    rows = [[row.comparison, decimal(row.gain_log_loss, 5),
             f"[{decimal(row.conditional_low,5)}; {decimal(row.conditional_high,5)}]",
             f"[{decimal(row.cluster_low,5)}; {decimal(row.cluster_high,5)}]"]
            for row in table.itertuples()]
    write_table("model_comparisons", ["Сравнение", r"$\Delta L$", "Компании", "Наборы"], rows,
                r"@{}>{\raggedright\arraybackslash}p{6.0cm}rrr@{}")


def mean_pr_curve(predictions, recall_grid):
    curves, weights = [], []
    for _, part in predictions.groupby(["repeat", "fold"]):
        precision, recall, _ = precision_recall_curve(part.y, part.probability)
        r, p = recall[::-1], precision[::-1]
        curves.append(p[np.minimum(np.searchsorted(r, recall_grid, side="left"), len(r)-1)])
        weights.append(len(part))
    return np.average(curves, axis=0, weights=weights)


def latex_escape(text):
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def write_table(name, header, rows, columns):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def extended_reports(metrics, comparisons, protocol):
    grouped = metrics[metrics.scheme.eq("grouped")].set_index("model")
    repeats = pd.read_csv(OUT / "metrics_by_repeat.csv")
    diagnostic = pd.read_csv(OUT / "split_diagnostics.csv")
    rows = []
    for scheme, label in [("grouped", "По наборам"), ("stratified", "Случайное")]:
        for model in ["constant", "lr_tags", "lr_all"]:
            d = repeats[repeats.scheme.eq(scheme) & repeats.model.eq(model)]
            rows.append([label, MODEL_NAMES[model], decimal(d.log_loss.mean()),
                         f"[{decimal(d.log_loss.min())}; {decimal(d.log_loss.max())}]",
                         f"[{decimal(d.roc_auc.min())}; {decimal(d.roc_auc.max())}]"])
    write_table("repeat_ranges", ["Схема", "Модель", "LogLoss", "Размах LogLoss", "Размах AUC"], rows,
                r"@{}l>{\raggedright\arraybackslash}p{4.4cm}rrr@{}")
    rows = []
    for experiment, label in [("main", "Основная"), ("without_ik12", "Без IK12"), ("acquisitions_only", "Без публичных")]:
        folder = OUT if experiment == "main" else OUT / experiment
        result = json.loads((folder / "results.json").read_text())
        table = pd.read_csv(folder / "metrics.csv")
        table = table[table.scheme.eq("grouped")].set_index("model")
        comp = pd.read_csv(folder / "comparisons.csv")
        comp = comp[comp.scheme.eq("grouped")].set_index(["new", "reference"])
        for model in ["lr_team", "lr_tags", "lr_all"]:
            gain = comp.loc[(model, "lr_base" if model == "lr_team" else "lr_team")]
            rows.append([label, str(result["n_known"]), MODEL_NAMES[model], decimal(table.loc[model,"log_loss"]),
                         decimal(table.loc[model,"average_precision"]), decimal(gain.gain_log_loss,5),
                         f"[{decimal(gain.cluster_low,5)}; {decimal(gain.cluster_high,5)}]"])
    write_table("sensitivity", ["Выборка", "$n$", "Модель", "LogLoss", "AP", r"$\Delta L$", r"Наборы: 95\%"], rows,
                r"@{}l r >{\raggedright\arraybackslash}p{3.2cm}rrrr@{}")
    ablation = ["lr_all", "lr_no_tags", "lr_no_text", "lr_no_geo", "lr_no_team"]
    write_table("ablation", ["Модель", "LogLoss", "AP", "ROC-AUC"],
                [[MODEL_NAMES[m]]+[decimal(grouped.loc[m,c]) for c in ["log_loss","average_precision","roc_auc"]] for m in ablation],
                r"@{}>{\raggedright\arraybackslash}p{8cm}rrr@{}")
    tags = pd.read_csv(OUT / "tag_summary.csv").head(12)
    write_table("tag_coefficients", ["Метка", "$n$", "$n_+$", r"Медиана $\beta$", r"10--90\%", r"$\beta>0$, \%"],
                [[latex_escape(r.tag),str(r.n),str(r.positive),decimal(r.median,3),
                  f"[{decimal(r.q10,3)}; {decimal(r.q90,3)}]", decimal(r.positive_fraction*100,0)] for r in tags.itertuples()],
                r"@{}>{\raggedright\arraybackslash}p{4.6cm}rrrrr@{}")
    fig, ax = canvas("Коэффициенты частых тегов в логистической модели", protocol, (12, 8))
    for i,r in enumerate(tags.itertuples()):
        ax.plot([r.q10,r.q90],[i,i],color=BLUE,linewidth=2)
        ax.scatter(r.median,i,color=DARK,s=30)
    ax.set_yticks(range(len(tags)), tags.tag, fontsize=10); ax.invert_yaxis()
    ax.axvline(0,color="gray",linestyle="--"); ax.set_xlabel("Коэффициент при наличии метки; логарифм шансов")
    save(fig,"07_tag_coefficients","12 самых частых меток в выборке с известным исходом. Медиана и 10–90%-ный диапазон коэффициентов по 25 групповым обучающим частям; это не доверительные интервалы и не причинные эффекты.")
    # Empirical heterogeneity across batches for the principal comparison.
    batches = pd.read_csv(OUT / "gains_by_batch.csv")
    b = batches[batches.scheme.eq("grouped") & batches.comparison.eq("Тематические теги")].sort_values("batch")
    fig,ax=canvas("Добавление тегов: различия между наборами YC",protocol,(12,9))
    fig.subplots_adjust(left=.13)
    ax.scatter(b.gain_log_loss,range(len(b)),s=np.maximum(10,np.sqrt(b.n)*4),color=BLUE)
    ax.set_yticks(range(len(b)),[f"{r.batch} (n={r.n})" for r in b.itertuples()],fontsize=8)
    ax.axvline(0,color="gray",linestyle="--");ax.invert_yaxis();ax.set_xlabel("Уменьшение LogLoss относительно модели с командой")
    save(fig,"08_batch_gains","Средняя парная разность пяти групповых повторов внутри набора. Размер точки связан с числом компаний. Малые наборы дают особенно нестабильные оценки; точки не являются независимыми повторными экспериментами.")
    values = {}
    names = {"constant":"Constant", "lr_base":"Base", "lr_team":"Team", "lr_tags":"Tags", "lr_all":"Full", "lr_all_ap":"FullAP", "lr_all_balanced":"Balanced", "cb_team":"Boost"}
    for m,label in names.items():
        for col,suffix in [("log_loss","Loss"),("average_precision","AP"),("roc_auc","AUC"),("precision_at_10","Precision"),("recall_at_10","Recall"),("lift_at_10","Lift")]:
            values[label+suffix]=decimal(grouped.loc[m,col],4)
        values[label+"PrecisionPercent"]=decimal(grouped.loc[m,"precision_at_10"]*100,2)
    values["RelativeGain"]=decimal(100*(grouped.loc["constant","log_loss"]-grouped.loc["lr_all","log_loss"])/grouped.loc["constant","log_loss"],2)
    d=diagnostic[diagnostic.scheme.eq("grouped")].adjusted_rand
    values["SplitMin"],values["SplitMax"]=decimal(d.min(),3),decimal(d.max(),3)
    comp=comparisons[comparisons.scheme.eq("grouped")].set_index(["new","reference"])
    for pair,label in [(("lr_tags","lr_team"),"TagGain"),(("lr_all","lr_tags"),"FullOverTags"),(("lr_all","constant"),"FullGain")]:
        r=comp.loc[pair]
        values[label]=decimal(r.gain_log_loss,5)
        values[label+"Low"],values[label+"High"]=decimal(r.cluster_low,5),decimal(r.cluster_high,5)
    values["PositiveBatches"]=str(int(b.gain_log_loss.gt(0).sum())); values["BatchCount"]=str(len(b))
    (TABLES / "model_numbers.tex").write_text("\n".join(r"\newcommand{\res"+key+"}{"+value+"}" for key,value in values.items())+"\n")
    tag = comp.loc[("lr_tags", "lr_team")]
    full = comp.loc[("lr_all", "lr_tags")]
    interpretation = [
        "Добавление тематических меток изменило LogLoss относительно модели с командой на "
        + decimal(tag.gain_log_loss, 5) + " в сторону уменьшения. "
        + ("Условный диапазон повторной выборки наборов целиком находится выше нуля. "
           if tag.cluster_low > 0 else "Условный диапазон повторной выборки наборов включает ноль либо отрицательные значения. ")
        + "Это оценка блока признаков в выбранной процедуре обучения, а не доказательство причинного действия тегов.",
        ("Преимущество полной модели перед моделью с тегами не установлено устойчиво: условный диапазон по наборам включает ноль."
         if full.cluster_low <= 0 <= full.cluster_high else
         "Разность полной модели и модели с тегами сохраняет знак в условном диапазоне по наборам; эта оценка не учитывает переобучение и требует независимой проверки.")
    ]
    signs = []
    for name in ["without_ik12", "acquisitions_only"]:
        table = pd.read_csv(OUT / name / "comparisons.csv")
        gain = table.loc[table.new.eq("lr_tags") & table.reference.eq("lr_team"), "gain_log_loss"].iloc[0]
        signs.append(gain > 0)
    interpretation.append(
        "Среднее улучшение от добавления тегов сохраняется при исключении IK12 и публичных компаний. "
        if all(signs) else "Направление среднего изменения от тегов зависит от состава дополнительной выборки. ")
    interpretation[-1] += "Проверки чувствительности используют ту же исходную базу и не являются внешней валидацией."
    direction_ap = "выше" if grouped.loc["lr_all_balanced", "average_precision"] > grouped.loc["lr_all_ap", "average_precision"] else "ниже"
    direction_p = "выше" if grouped.loc["lr_all_balanced", "precision_at_10"] > grouped.loc["lr_all_ap", "precision_at_10"] else "ниже"
    interpretation.append(f"При одинаковом подборе по AP взвешенная модель имеет среднюю AP {direction_ap}, "
                          f"а точность верхней десятой части {direction_p}, чем вариант без весов. "
                          "Эти наблюдаемые различия не трактуются как доказательство универсального эффекта взвешивания.")
    (TABLES / "interpretation.tex").write_text("\n\n".join(interpretation)+"\n", encoding="utf-8")
    positive = tags.loc[tags.positive_fraction.eq(1)].head(2)
    negative = tags.loc[tags.positive_fraction.eq(0)].head(2)
    tag_text = []
    for table, direction in [(positive, "положительный"), (negative, "отрицательный")]:
        if len(table):
            names = ", ".join(latex_escape(r.tag) + " (медиана " + decimal(r.median, 3) + ")"
                              for r in table.itertuples())
            tag_text.append(f"Метки {names} имеют {direction} коэффициент во всех 25 обучающих частях.")
    tag_text.append("Различия знаков у близких по смыслу обозначений, например AI и Artificial Intelligence, "
                    "дополнительно показывают зависимость интерпретации от таксономии и сочетания меток. "
                    "Изменение формулировки карточки не является установленным способом изменить исход компании.")
    (TABLES / "tag_interpretation.tex").write_text(" ".join(tag_text)+"\n", encoding="utf-8")


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
    code_path = config.PROJECT_ROOT / "src" / "modeling.py"
    if hashlib.sha256(code_path.read_bytes()).hexdigest() != protocol["code_sha256"]:
        raise RuntimeError("Код моделирования изменился после обучения")
    metrics = pd.read_csv(OUT / "metrics.csv")
    comparisons = pd.read_csv(OUT / "comparisons.csv")
    # Preserve ties and ordering of nearly equal probabilities after CSV round trip.
    predictions = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
    shap = pd.read_csv(OUT / "shap_oof.csv")
    suite = json.loads((OUT / "suite_complete.json").read_text())
    if suite["code_sha256"] != protocol["code_sha256"] or suite["completed_at"] < protocol["created_at"]:
        raise RuntimeError("Не завершены проверки чувствительности")
    metrics_plot(metrics, protocol)
    comparisons_plot(comparisons, protocol)
    calibration_plot(predictions, protocol)
    precision_recall_plot(predictions, metrics, protocol)
    top_decile_plot(metrics, protocol)
    shap_plot(shap, protocol)
    export_tables(metrics, comparisons)
    extended_reports(metrics, comparisons, protocol)
    print(f"Подготовлены 8 рисунков и таблицы: {FIGURES}")


if __name__ == "__main__":
    main()
