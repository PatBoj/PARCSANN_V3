from dataclasses import dataclass
from typing import List
import yaml
from pathlib import Path
from dataclasses import dataclass, field, InitVar


@dataclass
class InputFileConfig:
    data_dir: InitVar[Path]
    file_name: InitVar[str]

    file_path: Path = field(init=False)

    sheet_name: str | None = None
    keep_columns: str | None = None
    create_single_columns: dict | None = None
    create_multiple_columns: dict | None = None

    def __post_init__(self, data_dir: Path, file_name: str):
        self.file_path = (data_dir / file_name).resolve()

        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")


@dataclass
class ProjectConfig:
    name: str
    seed: int
    output_dir: str


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    target_column: str
    date_column: str


@dataclass
class FeatureConfig:
    lag_features: List[int]
    rolling_windows: List[int]
    use_holidays: bool


@dataclass
class ModelParams:
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float


@dataclass
class ModelConfig:
    type: str
    params: ModelParams


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    early_stopping_rounds: int


@dataclass
class EvaluationConfig:
    metric: str
    backtest_folds: int


@dataclass
class Config:
    project: ProjectConfig
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(**raw["data"]),
        features=FeatureConfig(**raw["features"]),
        model=ModelConfig(
            type=raw["model"]["type"],
            params=ModelParams(**raw["model"]["params"]),
        ),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
