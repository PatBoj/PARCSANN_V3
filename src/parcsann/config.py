from typing import List, Literal, Optional
import yaml
from pathlib import Path
from pydantic import BaseModel, confloat

CONFIG_DIR = Path(__file__).resolve().parent


class InputFileConfig(BaseModel):
    file_name: str
    sheet_name: Optional[str] = None
    keep_columns: List[str] = None
    create_single_columns: Optional[dict] = None
    create_multiple_columns: Optional[dict] = None

    file_path: Optional[Path] = None

    def resolve_path(self, base_dir: Path):
        self.file_path = (base_dir / self.file_name).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")


class ParcsannConfig(BaseModel):
    input_data_dir: Path

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
        self.input_data_dir = self.input_data_dir.resolve()
        print(self.input_data_dir)

        for file_cfg in [
            self.input_output_file_details,
            self.monocore_file_details,
            self.monocore_evolution_file_details,
        ]:
            file_cfg.resolve_path(self.input_data_dir)


def load_config(config_path: Path | None = None) -> ParcsannConfig:
    if not config_path:
        config_path = Path(__file__).resolve().parents[2]
        config_path = config_path / "configs" / "config_default.yaml"

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return ParcsannConfig(**raw)
