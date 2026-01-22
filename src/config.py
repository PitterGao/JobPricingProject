from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    models: Path = root / "models"


@dataclass(frozen=True)
class DataConfig:
    seed: int = 42
    n_companies: int = 500
    n_jobs: int = 1500
    n_days_logs: int = 21


@dataclass(frozen=True)
class TierConfig:
    weights: dict = None

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(self, "weights", {
                "training_budget": 0.25,
                "promotion_rate": 0.20,
                "work_life_balance": 0.20,
                "hr_response_speed": 0.15,
                "benefits_score": 0.20,
            })


@dataclass(frozen=True)
class FeatureConfig:
    categorical_cols: tuple = (
        "job_function",
        "job_level",
        "location",
        "funding_stage",
        "enterprise_tier",
    )
    numeric_cols: tuple = (
        "salary_min",
        "salary_max",
        "salary_mid",
        "company_size",
        "brand_level",
        "intl_flag",
        "target_top10",
        "talent_care_index",
        "apply_cnt",
        "topschoolratio",
        "expected_applies",
        "pred_impressions",
        "job_health_score",
    )


@dataclass(frozen=True)
class XGBConfig:
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            object.__setattr__(self, "params", {
                "n_estimators": 2000,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "min_child_weight": 2,
                "random_state": 42,
                "tree_method": "hist",
                "objective": "reg:squarederror"
            })


@dataclass(frozen=True)
class MLPConfig:
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 80


@dataclass(frozen=True)
class HealthConfig:
    total_days: int = 28
    history_len: int = 14
    horizon: int = 7
    feat_dim: int = 3
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
