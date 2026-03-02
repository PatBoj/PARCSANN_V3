from typing import List, Literal, Optional, Tuple
import yaml
from pathlib import Path
from pydantic import BaseModel, confloat

from parcsann.utils.dir import get_project_root
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(get_project_root(), ".env"))
ENV = os.getenv("ENV", "DEV")

CONFIG_DIR = Path(__file__).resolve().parent

# ======================================================================================================================
# MAIN CONFIG
# ======================================================================================================================


class InputFileConfig(BaseModel):
    file_name: str
    sheet_name: Optional[str] | None = None
    keep_columns: List[str] | None = None
    create_single_columns: Optional[dict] | None = None
    create_multiple_columns: Optional[dict] | None = None

    file_path: Optional[Path] | None = None

    def resolve_path(self, base_dir: Path):
        self.file_path = (base_dir / self.file_name).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")


class ParcsannConfig(BaseModel):
    project_root_dir: Path
    input_data_dir: str

    input_output_file_details: InputFileConfig
    monocore_file_details: InputFileConfig
    monocore_evolution_file_details: InputFileConfig

    use_normalization_layer: bool
    use_one_hot_encoding: bool
    use_monocores: bool
    train_split_ratio: confloat(gt=0, lt=1)

    input_columns: List[
        Literal[
            "cycle_length_in_days",
            "keff_max",
            "pxy_max",
            "pz_max",
            "keff_start",
            "ppf_start",
            "ppf_max",
            "ppf_end",
            "rho_start",
            "rho_max",
            "keff_evolution",
            "rho_evolution",
            "ppf_evolution",
        ]
    ]

    output_columns: List[
        Literal[
            "keff_start",
            "keff_max",
            "ppf_start",
            "ppf_max",
            "ppf_end",
            "cycle_length_in_days",
            "rho_start",
            "rho_max",
            "keff_evolution",
            "rho_evolution",
        ]
    ]

    def model_post_init(self, __context=None):
        self.input_data_dir = self.project_root_dir / self.input_data_dir

        for file_cfg in [
            self.input_output_file_details,
            self.monocore_file_details,
            self.monocore_evolution_file_details,
        ]:
            file_cfg.resolve_path(self.input_data_dir)


def load_config(config_path: Path | None = None) -> ParcsannConfig:
    config_suffix = "" if ENV == "PROD" else f"_{ENV}"
    
    if config_path:
        config_path = get_project_root() / config_path
    else:
        config_path = get_project_root() / f"configs{config_suffix}" / "config_default.yaml"

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return ParcsannConfig(project_root_dir=get_project_root(), **raw)


# ======================================================================================================================
# TUNE HYPERPARAMETERS CONFIG
# ======================================================================================================================


class TuneHyperparametersConfig(BaseModel):
    neurons: Tuple[int, int]
    activation: List[str]
    optimizer: List[
        Literal[
            "Adam",
            "RMSprop",
            "Adadelta",
            "Adamax",
            "Nadam",
            "Ftrl",
            "SGD",
        ]
    ]
    learning_rate: Tuple[float, float]
    decay_steps: Tuple[int, int]
    decay_rate: Tuple[float, float]
    layers_before_dropout: Tuple[int, int]
    layers_after_dropout: Tuple[int, int]
    dropout: Tuple[int, int]
    dropout_rate: Tuple[float, float]

    number_of_folds: int
    bayesian_inital_points: int
    bayesian_number_of_iterations: int
    train_split_ratio: float
    epochs: int

    def model_post_init(self, __context=None):
        self.neurons = tuple(self.neurons)
        self.learning_rate = tuple(self.learning_rate)
        self.decay_steps = tuple(self.decay_steps)
        self.decay_rate = tuple(self.decay_rate)
        self.layers_before_dropout = tuple(self.layers_before_dropout)
        self.layers_after_dropout = tuple(self.layers_after_dropout)
        self.dropout = tuple(self.dropout)
        self.dropout_rate = tuple(self.dropout_rate)


def load_tune_hyperparameters_config(config_path: Path | None = None) -> TuneHyperparametersConfig:
    config_suffix = "" if ENV == "PROD" else f"_{ENV}"

    if config_path:
        config_path = get_project_root() / config_path
    else:
        config_path = get_project_root() / f"configs{config_suffix}" / "config_tune_hyperparameters.yaml"

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return TuneHyperparametersConfig(project_root_dir=get_project_root(), **raw)