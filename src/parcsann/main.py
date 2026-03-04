from parcsann.hyperparameters.tune_hyperparametes import TuneHyperparametes
from parcsann.read_data.nuclear_code_data import CoreData
from parcsann.config import load_config
from pathlib import Path

config_dir = Path("configs/experiment_1")

for cfg_path in [config_dir / f"config_{i}.yaml" for i in range (1, 7)]:
    config = load_config(cfg_path)

    core_data = CoreData(config)
    core_data.process_data()

    tune_hyperparameters = TuneHyperparametes(core_data, experiment_dir="experiment_1")
    tune_hyperparameters.run(config)