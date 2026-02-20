from parcsann.read_data.nuclear_code_data import CoreData
from parcsann.config import load_tune_hyperparameters_config
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from pathlib import Path

from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from loguru import logger
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Normalization
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from keras.optimizers.schedules import ExponentialDecay

from math import floor
from sklearn.metrics import make_scorer, mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold


class TuneHyperparametes:
    optimizer_map = {
        "Adam": Adam,
        "RMSprop": RMSprop,
        "Adadelta": Adadelta,
        "Adagrad": Adagrad,
        "Adamax": Adamax,
        "Nadam": Nadam,
        "Ftrl": Ftrl,
        "SGD": SGD,
    }

    def __init__(self, core_data: CoreData, hyperparameters_config_path: Path | None = None):
        self.config = load_tune_hyperparameters_config(hyperparameters_config_path)

        self.X_train, self.X_val, self.y_train, self.y_val = core_data.train_test_div()
        
        
    def run_bayesian_optimization(self, f: Callable[..., float] | None):
        nn_parameters = {
            'neurons': self.config.neurons,
            'activation': (0, 5),
            'optimizer': (0, 6),
            'learning_rate': self.config.learning_rate,
            'layers1': self.config.layers_before_dropout,
            'layers2': self.config.layers_after_dropout,
            'dropout': self.config.dropout,
            'dropout_rate': self.config.dropout_rate,
            'decay_steps': self.config.decay_steps,
            'decay_rate': self.config.decay_rate,
        }

        nn_best_parameters = BayesianOptimization(f, nn_parameters)
        nn_best_parameters.maximize(init_points=25, n_iter=40)

        return 