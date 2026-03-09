import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from parcsann.read_data.nuclear_code_data import CoreData
from parcsann.config import load_tune_hyperparameters_config
from parcsann.config import load_config
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from pathlib import Path
from typing import Callable, Self
from dataclasses import asdict
from parcsann.config import ParcsannConfig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from loguru import logger

from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Dropout, Normalization
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from keras.optimizers.schedules import ExponentialDecay

from functools import cached_property
import json

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from datetime import datetime

from parcsann.utils.dir import get_project_root
from parcsann.config import ENV


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

    def __init__(
            self, 
            core_data: CoreData, 
            hyperparameters_config_path: Path | None = None, 
            experiment_dir: str = None,
        ) -> None:
        self.config = load_tune_hyperparameters_config(hyperparameters_config_path)
        self.activation_naming_map = dict(enumerate(self.config.activation))
        self.optimizer_naming_map = dict(enumerate(self.config.optimizer))

        self.X_train, self.X_test, self.y_train, self.y_test = core_data.train_test_div(self.config.train_split_ratio)
        if ENV == "DEV":
            self.X_train = self.X_train[:19]
            self.X_test = self.X_test[:2]
            self.y_train = self.y_train[:19]
            self.y_test = self.y_test[:2]

        self.bayesian_optimization: BayesianOptimization | None = None

        output_dir_suffix = "" if ENV == "PROD" else f"_{ENV}"
        
        self.output_dir = get_project_root() / f"hyperparameters{output_dir_suffix}"
        if experiment_dir:
            self.output_dir = self.output_dir / experiment_dir
        self.output_dir = self.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_and_compute_nn(
            self, neurons: float, activation: float, optimizer: float, learning_rate: float, normalize: float,
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
            nn.add(Input(shape=(self.X_train.shape[1],)))

            for _ in range(layers_before_dropout):
                nn.add(Dense(neurons, activation=activation))
            
            if dropout:
                nn.add(Dropout(dropout_rate))
    
            for _ in range(layers_after_dropout):
                nn.add(Dense(neurons, activation=activation))
            
            nn.add(Dense(self.y_train.shape[1], activation="linear"))
            nn.compile(loss="mse", optimizer=optimizer_instance)
            return nn
        
        es = EarlyStopping(monitor="loss", mode="min", verbose=0, patience=10, restore_best_weights=True)
        nn = KerasRegressor(model=build_nn, epochs=self.config.epochs, verbose=0, fit__callbacks=[es])

        kfold = KFold(n_splits=self.config.number_of_folds, shuffle=True)

        if normalize > 0.5:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("nn", nn),
            ])
        else:
            model = nn
        
        score = cross_val_score(model, self.X_train, self.y_train, scoring="neg_mean_squared_error", cv=kfold)
        if np.isnan(score).any():
            return -1e10

        return score.mean()
        
    def run_bayesian_optimization(self, f: Callable[..., float] | None) -> Self:
        nn_parameters = {
            "neurons": self.config.neurons,
            "activation": (0, len(self.activation_naming_map) - 1),
            "optimizer": (0, len(self.optimizer_naming_map) - 1),
            "learning_rate": self.config.learning_rate,
            "normalize": self.config.normalize,
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
        bool_values = ["dropout", "normalize"]
        
        trans_params = {}
    
        for key, value in params.items():
            if key in int_values:
                trans_params[key] = int(round(value))
            elif key in bool_values:
                trans_params[key] = bool(round(value))

            elif key == "activation":
                trans_params[key] = self.activation_naming_map[round(value)]
            elif key == "optimizer":
                trans_params[key] = self.optimizer_naming_map[round(value)]
            else:
                trans_params[key] = value

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
            json.dump(self.get_all_parameters, f, indent=4)
    
    @cached_property
    def get_best_parameters(self):
        return self.transform_parameters(self.bayesian_optimization.max["params"])

    def save_best_parameters(self):
        with open(self.output_dir / "best_parameters.json", "w") as f:
            json.dump(self.get_best_parameters, f, indent=4)

    def linear_regression(self):
        linear_regression = LinearRegression()
        linear_regression.fit(self.X_train, self.y_train)
        y_pred = linear_regression.predict(self.X_test)

        return {
            # "model": nn,
            "y_pred": y_pred,
            "mse": mean_squared_error(self.y_test, y_pred),
            "mape": mean_absolute_percentage_error(self.y_test, y_pred),
        }
    
    def neural_network(self):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.get_best_parameters["learning_rate"],
            decay_steps=self.get_best_parameters["decay_steps"],
            decay_rate=self.get_best_parameters["decay_rate"],
            staircase=False
        )

        optimizer_instance = self.optimizer_map[self.get_best_parameters["optimizer"]](learning_rate=lr_schedule)

        def build_nn():
            nn = Sequential()
            nn.add(Input(shape=(self.X_train.shape[1],)))

            for _ in range(self.get_best_parameters["layers_before_dropout"]):
                nn.add(Dense(
                    self.get_best_parameters["neurons"],
                    activation=self.get_best_parameters["activation"]
                ))

            if self.get_best_parameters["dropout"]:
                nn.add(Dropout(self.get_best_parameters["dropout_rate"]))

            for _ in range(self.get_best_parameters["layers_after_dropout"]):
                nn.add(Dense(
                    self.get_best_parameters["neurons"],
                    activation=self.get_best_parameters["activation"]
                ))

            nn.add(Dense(self.y_train.shape[1], activation="linear"))
            nn.compile(loss="mse", optimizer=optimizer_instance)

            return nn
        
        es = EarlyStopping(monitor="loss", mode="min", verbose=0, patience=10, restore_best_weights=True)   
        nn = KerasRegressor(
            model=build_nn,
            epochs=self.config.epochs,
            callbacks=[es],
            verbose=0
        )

        if self.get_best_parameters["normalize"]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("nn", nn),
            ])
            model.fit(self.X_train, self.y_train, nn__callbacks=[es])
        else:
            model = nn
            model.fit(self.X_train, self.y_train, callbacks=[es])
        
        y_pred = model.predict(self.X_test)

        return {
            # "model": nn,
            "y_pred": y_pred,
            "mse": mean_squared_error(self.y_test, y_pred),
            "mape": mean_absolute_percentage_error(self.y_test, y_pred),
        }
    
    def create_and_save_plot(self, y_pred_nn, y_pred_lr):
        df = pd.DataFrame({
            "y_test": self.y_test.flatten(),
            "y_pred_nn": y_pred_nn,
            "y_pred_lr": y_pred_lr,
        })

        df.to_csv(self.output_dir / "predictions.csv", index=False)
        
        df["rad_nn"] = np.abs(df["y_test"] - df["y_pred_nn"]) / (np.abs(df["y_test"]))
        df["rad_lr"] = np.abs(df["y_test"] - df["y_pred_lr"]) / (np.abs(df["y_test"]))

        global_min = min(df["rad_nn"].min(), df["rad_lr"].min())
        global_max = max(df["rad_nn"].max(), df["rad_lr"].max())
        bins = np.linspace(global_min, global_max, 36)

        plt.figure(figsize=(10, 6))
        plt.hist(df["rad_nn"], bins=bins, alpha=0.5, label="Neural Network")
        plt.hist(df["rad_lr"], bins=bins, alpha=0.5, label="Linear Model")
        plt.legend()
        plt.xlabel("Relative Absolute Difference")
        plt.savefig(self.output_dir / "comparison_histogram.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(
            df["y_test"], 
            df["y_pred_nn"], 
            linestyle='None',
            marker='o', 
            markerfacecolor='none',
            markeredgecolor='blue',
            label="Neural Network"
        )

        plt.plot(
            df["y_test"], 
            df["y_pred_lr"], 
            linestyle='None',
            marker='s', 
            markerfacecolor='none',
            markeredgecolor='red',
            label="Linear Model"
        )

        min_val = df["y_test"].min()
        max_val = df["y_test"].max()
        plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='y=x')

        plt.legend()
        plt.savefig(self.output_dir / "scatter_plot.png", bbox_inches="tight")
    
    def compare_scores(self):
        linear_output = self.linear_regression()
        nn_output = self.neural_network()

        self.create_and_save_plot(nn_output["y_pred"].flatten(), linear_output["y_pred"].flatten())

        msg = (
            f"Linear model mse: {linear_output["mse"]:.3g}, and neural network mse: {nn_output["mse"]:.3g}, "
            f"neural network mse is better: {100*(linear_output["mse"] - nn_output["mse"]) / linear_output["mse"]:.3g}%\n"
            f"Linear model mape: {100*linear_output["mape"]:.3g}%, and neural network mape: {100*nn_output["mape"]:.3g}%, "
            f"neural network mape is better: {100*(linear_output["mape"] - nn_output["mape"]) / linear_output["mape"]:.3g}%"
        )

        logger.info(msg)
        with open(self.output_dir / "comparison.txt", "a") as f:
            f.write(msg + "\n")

    def save_input_output_columns(self, parcasnn_config: ParcsannConfig | None = None):
        if not parcasnn_config:
            parcasnn_config = load_config()

        with open(self.output_dir / "input_output_columns.txt", "a") as f:
            f.write("INPUT COLUMNS:\n")
            f.write("\n".join(parcasnn_config.input_columns))
            f.write("\n\nOUTPUT COLUMNS:\n")
            f.write("\n".join(parcasnn_config.output_columns))
            f.write("\n\n")
            f.write(f"USE MONOCORES: {parcasnn_config.use_monocores}\n")
            f.write(f"USE ONE HOT ENCODING: {parcasnn_config.use_one_hot_encoding}\n")
            f.write("\n")

    # def save_read_model_dummy(self):
    #     import joblib

    #     joblib.dump(model, "model.pkl")
    #     model = joblib.load("model.pkl")
    #     y_pred = model.predict(X_new)

    def run(self, parcasnn_config: ParcsannConfig | None = None):
        self.run_bayesian_optimization(self.create_and_compute_nn)
        self.save_all_parameters()
        self.save_best_parameters()
        self.save_input_output_columns(parcasnn_config)
        self.compare_scores()


# I HAVE TO CREATE A FUNCTION TO COMPARE EVOLUTIONS