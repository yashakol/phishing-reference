"""Build human-readable results and static QA figures from measured outputs."""

import json
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"


def main():
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    audit = json.loads((REPORTS / "data_audit.json").read_text())
    submissions = json.loads((REPORTS / "submission_checks.json").read_text())
    config = metrics["selected_configuration"]
    holdout = metrics["untouched_holdout"]["primary_ensemble"]
    domain = metrics["unseen_domain_check"]["primary_ensemble"]
    rows = []
    for name, score in metrics["untouched_holdout"].items():
        rows.append({"check": "untouched_holdout", "model": name, **score})
    for name in ["primary_ensemble", "text", "boost"]:
        rows.append({"check": "unseen_domain", "model": name, **metrics["unseen_domain_check"][name]})
    pd.DataFrame(rows).to_csv(REPORTS / "metrics_table.csv", index=False)
    label_counts = audit["label_counts"]
    test = audit["test"]
    bundle = joblib.load(ROOT / "artifacts" / "primary.joblib")
    text = bundle["text"]
    feature_names = ["char:" + item for item in text.char.get_feature_names_out()]
    feature_names += ["token:" + item for item in text.word.get_feature_names_out()]
    weights = text.model.coef_.ravel()
    order = np.argsort(weights)
    inspect = np.concatenate([order[:100], order[-100:][::-1]])
    pd.DataFrame({"feature": [feature_names[int(i)] for i in inspect], "weight": weights[inspect]}).to_csv(REPORTS / "text_feature_weights.csv", index=False)
    del bundle, text, feature_names, weights
    gc.collect()
    importance = pd.read_csv(REPORTS / "feature_importance.csv")
    lines = [
        "# Результаты выполненного прогона", "",
        "Локальные метрики относятся к проверкам на размеченном train. Метки test неизвестны; F1 турнира не вычислялся.", "",
        "## Данные", "",
        f"- Пригодные записи до удаления повторов: {audit['accepted_before_deduplication']}.",
        f"- Исключено поврежденных записей: {audit['rejected_records']}.",
        f"- Удалено повторяющихся записей: {audit['duplicate_rows_removed']}.",
        f"- Противоречивых URL: {audit['conflicting_urls_removed']}.",
        f"- Очищенный train: {audit['clean_unique_rows']} уникальных URL; безопасных {label_counts['0']}, фишинговых {label_counts['1']}.",
        f"- Test: {test['rows']} строк; повторяющихся URL {test['duplicate_url_rows']}.",
        f"- В test совпадает с train: {test['known_exact_url_rows']} строк по полному URL, {test['known_host_rows']} по host и {test['known_registered_domain_rows']} по регистрируемому домену.",
        "", "Поврежденные строки исключены без попыток придумать их метки или восстановить потерянный фрагмент адреса.", "",
        "## Настройка модели", "",
        f"Выбраны текстовая модель `{config['text_model']}` и CatBoost с весом текста {config['text_weight']}. Порог основного ответа: {config['threshold']:.6f}.",
        f"Все параметры и пороги выбраны по трехкратной кросс-валидации на {metrics['split']['development_rows']} внутренних URL. Отложенные {metrics['split']['holdout_rows']} URL не участвовали в выборе параметров.",
        "", "## Проверки качества", "",
        "| Проверка | Модель | F1 класса 1 | Precision | Recall | Порог | Строк |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        check = "Контрольная выборка" if row["check"] == "untouched_holdout" else "Новые домены"
        lines.append(f"| {check} | {row['model']} | {row['f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['threshold']:.6f} | {row['rows']} |")
    lines += ["", "В проверке новых доменов число общих регистрируемых доменов между обучением и проверкой равно 0. Ее разбиение выполнено только внутри обучающих 80%, без использования контрольной выборки.", "",
              f"Основное решение на контрольной выборке: правильно найдены {holdout['tp']} фишинговых URL, пропущены {holdout['fn']}, ложных тревог {holdout['fp']}, правильно распознаны {holdout['tn']} безопасных URL.",
              f"Проверка новых доменов: правильно найдены {domain['tp']} фишинговых URL, пропущены {domain['fn']}, ложных тревог {domain['fp']}.", "",
              "Разница между проверками показывает ограничение переносимости на новые домены. Проверка новых доменов также имеет другую композицию адресов; сравнение не является доказательством одной-единственной причины разницы.", "",
              "![Проверки качества и матрица ошибок](validation.png)", "",
              "## Что использовала модель", "",
              "Наиболее важные структурные признаки CatBoost в обучении на внутренних 80%:", "",
              "| Признак | Важность |", "|---|---|",
              *[f"| `{row.feature}` | {row.importance:.3f} |" for row in importance.head(8).itertuples()], "",
              "Это относительная важность в конкретной обученной модели, а не доказательство причинной связи и не универсальное правило фишинга. Веса 100 наиболее положительных и 100 наиболее отрицательных текстовых признаков сохранены в `text_feature_weights.csv`.", "",
              "## Ответы для турнира", "",
              f"- `submission_primary.csv`: {submissions['primary']['rows']} строк, прогнозов фишинга {submissions['primary']['positive_predictions']}.",
              f"- `submission_text.csv`: {submissions['text']['rows']} строк, прогнозов фишинга {submissions['text']['positive_predictions']}.",
              f"- Число строк test с различающимся прогнозом: {submissions['different_predictions']}.",
              f"- Точная известная метка из train применена к {submissions['primary']['exact_known_url_overrides']} совпадающим URL в test.", "",
              "Основной вариант выбран по внутренней кросс-валидации. Второй является самостоятельной текстовой альтернативой. Сравнение на приватной таблице пока неизвестно.", "",
              "## Технические сведения", "",
              f"Полный прогон обучения и локальной оценки занял {metrics['total_training_seconds']:.1f} секунд в зафиксированном окружении. Время установки зависимостей сюда не входит.",
              "Версии библиотек находятся в `environment.json` и `../requirements.txt`. Индексы разбиений сохранены в NPZ; входные SHA-256 записаны в `data_audit.json`.", "",
              "Результаты тестов реализации сохранены в `tests.json`. Проверка разбиений, пересчет метрик и побайтовое повторение обоих ответов без сети описаны в `verification.json`.", ""]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    names = ["Текст", "CatBoost", "Ансамбль"]
    text_name = config["text_model"]
    h = [metrics["untouched_holdout"][text_name]["f1"], metrics["untouched_holdout"]["boost"]["f1"], holdout["f1"]]
    d = [metrics["unseen_domain_check"]["text"]["f1"], metrics["unseen_domain_check"]["boost"]["f1"], domain["f1"]]
    y = np.arange(3)
    for values, offset, color, label in [(h, -0.17, "#2166ac", "Случайная контрольная"), (d, 0.17, "#e08214", "Новые домены")]:
        bars = axes[0].barh(y + offset, values, height=0.3, color=color, label=label)
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10)
    axes[0].set_yticks(y, names)
    axes[0].set_xlim(0, 1.10)
    axes[0].set_xlabel("F1 для фишинга")
    axes[0].set_title("Качество на размеченных данных", pad=14)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.30), frameon=False, fontsize=9)
    matrix = np.array([[holdout["tn"], holdout["fp"]], [holdout["fn"], holdout["tp"]]])
    axes[1].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18,
                         color="white" if matrix[i, j] > matrix.max() * 0.5 else "#172e46")
    axes[1].set_xticks([0, 1], ["Безопасный", "Фишинг"])
    axes[1].set_yticks([0, 1], ["Безопасный", "Фишинг"])
    axes[1].set_xlabel("Прогноз модели")
    axes[1].set_ylabel("Истинный класс")
    axes[1].set_title("Контрольная выборка: ансамбль", pad=14)
    fig.tight_layout(pad=2)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(REPORTS / "validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Created reports/RESULTS.md and reports/validation.png")


if __name__ == "__main__":
    main()
