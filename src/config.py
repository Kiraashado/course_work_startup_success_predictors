from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    PREPROCESSED_DIR: Path = ARTIFACTS_DIR / "preprocessed"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    EDA_DIR: Path = ARTIFACTS_DIR / "eda" / "cohort_2023_2026"
    BASELINE_DATE: str = "2023-07-13"
    FOLLOWUP_DATE: str = "2026-08-01"
    COHORT_FIGURES_DIR: Path = PROJECT_ROOT / "coursework_text" / "graphics" / "eda"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    
    SOURCE_FILES: Dict[str, str] = field(default_factory=lambda: {
        "yc_2023_feb": "2023-02-27-yc-companies.csv",
        "yc_2023_jul": "2023-07-13-yc-companies.csv",
        "yc_2026": "yc_companies.csv",
    })

    SNAPSHOT_DATES: Dict[str, str] = field(default_factory=lambda: {
        "yc_2023_feb": "2023-02-27",
        "yc_2023_jul": "2023-07-13",
        "yc_2026": "2026-08-01",
    })

    COMPARE_COLS_2023: List[str] = field(default_factory=lambda: [
        "status", "team_size", "tags", "batch"
    ])
    
    UNIFIED_SCHEMA: List[str] = field(default_factory=lambda: [
        "id",                    # Уникальный идентификатор (company_id)
        "name",                  # Название компании
        "source",                # Источник данных
        "snapshot_date",         # Дата среза
        "status",                # Статус: active/inactive/acquired/public
        "short_description",     # Краткое описание
        "long_description",      # Полное описание
        "industry",              # Отрасль
        "subindustry",           # Подотрасль
        "tags",                  # Теги (список)
        "batch",                 # Батч YC (e.g., W22, S21)
        "year_founded",          # Год основания
        "team_size",             # Размер команды
        "num_founders",          # Количество основателей
        "founders_names",        # Имена основателей
        "country",               # Страна
        "city",                  # Город
        "location_raw",          # Сырое местоположение
        "website",               # Сайт
        "crunchbase_url",        # URL на Crunchbase
        "linkedin_url",          # URL на LinkedIn
        "top_company",           # является ли компания топовой
        "is_hiring",             # Занимается ли компания активным наймом
        "stage",                 # Стадия развития: Early/Growth
        "nonprofit",             # Является ли компания некоммерческой
    ])
    
    COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "yc_2023_feb": {
            "company_id": "id",
            "company_name": "name",
            "status": "status",
            "industry": "industry",
            "subindustry": "subindustry",
            "batch": "batch",
            "team_size": "team_size",
            "tags": "tags",
            "url": "website",
            "location": "location_raw",
            "cb_url": "crunchbase_url",
        },
        "yc_2023_jul": {
            "company_id": "id",
            "company_name": "name",
            "status": "status",
            "industry": "industry",
            "subindustry": "subindustry",
            "batch": "batch",
            "team_size": "team_size",
            "tags": "tags",
            "url": "website",
            "location": "location_raw",
            "cb_url": "crunchbase_url",
        },
        "yc_2026": {
            "id": "id",
            "name": "name",
            "one_liner": "short_description",
            "long_description": "long_description",
            "status": "status",
            "industry": "industry",
            "subindustry": "subindustry",
            "batch": "batch",
            "team_size": "team_size",
            "tags": "tags",
            "year_founded": "year_founded",
            "num_founders": "num_founders",
            "founders_names": "founders_names",
            "country": "country",
            "city": "city",
            "all_locations": "location_raw",
            "website": "website",
            "cb_url": "crunchbase_url",
            "linkedin_url": "linkedin_url",
            "top_company": "top_company", 
            "isHiring": "is_hiring",
            "stage": "stage", 
            "nonprofit": "nonprofit",
        },
    })
    
    STATUS_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "active": "active",
        "inactive": "inactive",
        "acquired": "acquired",
        "public": "public",
        "Active": "active",
        "Inactive": "inactive",
        "Acquired": "acquired",
        "Public": "public",
    })

    SUCCESS_STATUSES = {"acquired", "public"}
    FAILURE_STATUSES = {"inactive"}
    
    def __post_init__(self):
        """Создание необходимых директорий при инициализации."""
        for directory in [
            self.RAW_DATA_DIR, self.PREPROCESSED_DIR,
            self.REPORTS_DIR, self.EDA_DIR, self.FIGURES_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


config = Config()
