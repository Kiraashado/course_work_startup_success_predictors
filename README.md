# Анализ компаний Y Combinator

Текущий EDA описывает компании, активные 13 июля 2023 года, и их статусы
на 1 августа 2026 года. Все исходные признаки берутся из июльской выгрузки.
Февральская выгрузка используется только для проверки согласованности.
Отсутствующие в позднем срезе компании сохраняются с неизвестным исходом.

## Запуск

Из корня проекта в окружении с pandas, numpy, scipy, matplotlib и pyarrow:

```bash
.venv/bin/python src/data_preprocessing.py
.venv/bin/python src/eda_analysis.py
.venv/bin/python -m unittest discover -s tests -v
```

Пути исходных файлов и даты заданы в `src/config.py`.