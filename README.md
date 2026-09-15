# Анализ компаний Y Combinator

## Запуск

Из корня проекта в окружении с pandas, numpy, scipy, matplotlib и pyarrow:

```bash
.venv/bin/python src/data_preprocessing.py
.venv/bin/python src/eda_analysis.py
.venv/bin/python -m unittest discover -s tests -v
```

Пути исходных файлов и даты заданы в `src/config.py`.