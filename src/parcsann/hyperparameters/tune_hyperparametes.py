from parcsann.read_data.nuclear_code_data import CoreData
from parcsann.config import load_tune_hyperparameters_config
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from pathlib import Path

from typing import Callable, Self

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

from functools import cached_property
import json

from math import floor
from sklearn.metrics import make_scorer, mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from datetime import datetime

from parcsann.utils.dir import get_project_root


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

    def __init__(self, core_data: CoreData, hyperparameters_config_path: Path | None = None) -> None:
        self.config = load_tune_hyperparameters_config(hyperparameters_config_path)
        self.activation_naming_map = dict(enumerate(self.config.activation))
        self.optimizer_naming_map = dict(enumerate(self.optimizer_map))

        self.X_train, self.X_val, self.y_train, self.y_val = core_data.train_test_div(0.2)

        self.bayesian_optimization: BayesianOptimization | None = None

        self.output_dir = get_project_root() / "hyperparameters" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_and_compute_nn(
            self, neurons: float, activation: float, optimizer: float, learning_rate: float, 
            layers_before_dropout: float, layers_after_dropout: float, dropout: float, dropout_rate: float, 
            decay_steps: float, decay_rate: float
        ) -> float:        
        
        neurons = round(neurons)
        activation = self.activation_naming_map[round(activation)]
        layers_before_dropout = round(layers_before_dropout)
        layers_after_dropout = round(layers_after_dropout)
        decay_steps = round(decay_steps)

        def build_nn():
            lr_schedule = ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False
            )

            optimizer_instance = self.optimizer_map[self.optimizer_naming_map[round(optimizer)]](
                learning_rate=lr_schedule
            )

            nn = Sequential()
            nn.add(Dense(neurons, input_dim=self.X_train.shape[1], activation=activation))
            
            for _ in range(layers_before_dropout):
                nn.add(Dense(neurons, activation=activation))
            
            if dropout > 0.5:
                nn.add(Dropout(dropout_rate))
    
            for _ in range(layers_after_dropout):
                nn.add(Dense(neurons, activation=activation))
            
            nn.add(Dense(self.y_train.shape[1], activation="linear"))
            nn.compile(loss="mse", optimizer=optimizer_instance)
            return nn
        
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=10, restore_best_weights=True)
        nn = KerasRegressor(build_fn=build_nn, epochs=300, verbose=0, fit__callbacks=[es])

        kfold = KFold(n_splits=self.config.number_of_folds, shuffle=True)
        
        score = cross_val_score(nn, self.X_train, self.y_train, scoring="neg_mean_squared_error", cv=kfold)
        if np.isnan(score).any():
            return -1e10

        return score.mean()
        
    def run_bayesian_optimization(self, f: Callable[..., float] | None) -> Self:
        nn_parameters = {
            "neurons": self.config.neurons,
            "activation": (0, len(self.activation_naming_map)),
            "optimizer": (0, len(self.optimizer_naming_map)),
            "learning_rate": self.config.learning_rate,
            "layers_before_dropout": self.config.layers_before_dropout,
            "layers_after_dropout": self.config.layers_after_dropout,
            "dropout": self.config.dropout,
            "dropout_rate": self.config.dropout_rate,
            "decay_steps": self.config.decay_steps,
            "decay_rate": self.config.decay_rate,
        }

        self.bayesian_optimization = BayesianOptimization(f, nn_parameters)
        self.bayesian_optimization.maximize(
            init_points=self.config.bayesian_inital_points, n_iter=self.config.bayesian_number_of_iterations
        )

        return Self
    
    def transform_parameters(self, params: dict) -> dict:
        int_values = ["neurons", "layers_before_dropout", "layers_after_dropout", "decay_steps"]
        bool_values = ["dropout"]
        
        trans_params = {}
    
        for key, value in params.items():
            if key in int_values:
                replaced_value = int(round(value))
            elif key in bool_values:
                replaced_value = bool(round(value))

            elif key == "activation":
                replaced_value = self.activation_naming_map[round(value)]
            elif key == "optimization":
                replaced_value = self.optimizer_naming_map[round(value)]

            trans_params[key] = replaced_value

        return trans_params
    
    @cached_property
    def get_all_parameters(self):
        transformed_parameters = [
            {
                "target": res["target"],
                "params": self.transform_parameters(res["params"])
            }
            for res in self.bayesian_optimization.res
        ]

        return transformed_parameters

    def save_all_parameters(self):
        with open(self.output_dir / "all_parameters.json", "w") as f:
            json.dump(self.get_all_parameters(), f, indent=4)
    
    @cached_property
    def get_best_parameters(self):
        return self.transform_parameters(self.bayesian_optimization.max["params"])

    def save_best_parameters(self):
        with open(self.output_dir / "best_parameters.json", "w") as f:
            json.dump(self.get_best_parameters(), f, indent=4)

    def run(self):
        self.run_bayesian_optimization(self.create_and_compute_nn)
        self.save_all_parameters()
        self.save_best_parameters()
