from parcsann.hyperparameters.tune_hyperparametes import TuneHyperparametes
from parcsann.read_data.nuclear_code_data import CoreData

core_data = CoreData()
core_data.process_data()

tune_hyperparameters = TuneHyperparametes(core_data)
tune_hyperparameters.run()