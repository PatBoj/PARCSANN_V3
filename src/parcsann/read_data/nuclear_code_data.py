import re
import sys
from itertools import compress
from typing import Self
# from warnings import simplefilter

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from parcsann.config import ParcsannConfig, InputFileConfig

# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


## UTILS


def read_raw_file(file_path: str, sep: str | None = None, sheet_name: str | None = None) -> pd.DataFrame:
    """Reads raw .xslx or .csv file.

    Args:
        file_path (str): path to a file.
        sep (str, optional): data separator, used when reading .csv file. Defaults to `,`.
        sheet_name (str, optional): name of the sheet, used when reading .xlsx files. Defaults to `None`.

    Returns:
        pd.DataFrame: DataFrame of read file.
    """
    file_extension = file_path.split(".")[-1]

    logger.info(f"Reading a file: {file_path}.")
    if file_extension == "xlsx":
        return pd.read_excel(file_path, sheet_name)
    if file_extension == "csv":
        return pd.read_csv(file_path, sep)

    logger.error(f"Invalid file format: `{file_extension}`, only `csv` and `xlsx` are possible.")
    sys.exit(1)


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame read from an Excel file.

    Args:
        df (pd.DataFrame): a DataFrame with invalid column names.

    Returns:
        pd.DataFrame: a DataFrame with fixed column names.
    """
    logger.info("Fixing colum names.")
    good_col_names = [col.replace("\xa0", "") for col in df.columns]
    good_col_names = [col.strip() for col in good_col_names]
    good_col_names = [col.lower() for col in good_col_names]

    df.columns = good_col_names

    return df


def filter_columns(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    """Filters DataFrame by keeping columns specified in the `keep_cols`.

    Args:
        df (pd.DataFrame): a DataFrame to filter.
        keep_cols (list): list of column names that will remain in the DataFrame.

    Returns:
        pd.DataFrame: filtered DataFrame.
    """
    if not all(column in df.columns for column in keep_cols):
        logger.error(
            f"Columns {set(keep_cols) - set(df.columns)} are given in the `keep_cols`, "
            "but they are not in the dataframe.",
        )
        sys.exit(1)

    logger.info(f"From a dataframe keeping only `{keep_cols}` columns.")
    return df[keep_cols]


def create_series(df: pd.DataFrame, new_col_template: str, formula_template: str) -> pd.DataFrame:
    """Creates multiple columns with names based on the `new_col_template` and calculated using `formula_template`.
    For example if we want to change seconds to minutes in our data. Let's assume that we have multiple columns
    named `["sec1", "sec2", "sec3", ...]`. Using `new_col_template = "minN"` and `formula_template = "df.secN/60"`
    we create new columns: `["min1", "min2", "min3", ...]` where values are corresponding to seconds and divided by 60.
    We don't have to know total number of columns. Also it is possible to use more than one column in the formula.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        new_col_template (str): new column names template, it must contain `N` somewhere.
        formula_template (str): formula template that will be used to create new columns, it must contain `N` somewhere.

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if "N" not in new_col_template:
        logger.error(f"Incorrect new column naming template, this `{new_col_template}` must contain `N`.")
        sys.exit(1)

    if "N" not in formula_template:
        logger.error(f"Incorrect formula template, this `{formula_template}` must contain `N`.")
        sys.exit(1)

    logger.info(f"Creating new multiple columns: `{new_col_template}` using formula `{formula_template}`.")
    idx = 0

    while True:
        idx += 1
        new_col_name = new_col_template.replace("N", str(idx))
        formula = formula_template.replace("N", str(idx))

        if idx == 1:
            try:
                eval(formula)
            except AttributeError:
                logger.error(logger.error(f"Check if the formula is correct: `{formula}`."))
                sys.exit(1)

        try:
            df[new_col_name] = eval(formula)
        except AttributeError:
            break

    logger.info(f"Created {idx - 1} new columns `{new_col_template} = {formula_template}`.")
    return df


def create_multiple_cols(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Creates multiple series of new columns based on the dictionary. For detailed documentation look for
    `create_series` function that is used in this one.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        cfg (dict): configuration dictionary with new column names template and formula template.

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if cfg is None:
        logger.warning(
            "`multiple_cols` keyword was given in the configuration file, but it's empty. Proceeding without changes.",
        )
        return df

    for new_col, formula in cfg.items():
        df = create_series(df, new_col, formula)

    return df


def create_single_cols(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create a single column based on the template given in the configuration file.
    For example if we want to add a frequency to data that contains period, in the configuration file,
    we need to set `cfg={"frequency":"1/period"}`. Column named "period" must exist in the `df`.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        cfg (dict): dictionary that contains {new_column_name}:{formula}

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if cfg is None:
        logger.warning(
            "`single_cols` keyword was given in the configuration file, but it's empty. Proceeding without changes.",
        )
        return df

    for new_col, formula in cfg.items():
        logger.info(f"Creating a new columns: `{new_col}` based on the formula `{formula}`.")
        try:
            df[new_col] = eval(formula)
        except AttributeError:
            logger.error(f"Check if the formula is correct: `{formula}`.")
            sys.exit(1)

    return df


def create_new_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Creates new columns based on the given formulas. It is possible to create or more single columns using
    keyword `single_cols`. It is also possible to add multiple series of `N` new columns using keyword `multiple_cols`.
    How does each of them work look for `create_single_cols` and `create_series` functions.

    Args:
        df (pd.DataFrame): a DataFrame to which a new column is added.
        cfg (dict): configuration file that contains keywords `single_cols` and/or `multiple_cols`.

    Returns:
        pd.DataFrame: a DataFrame with added columns.
    """
    if cfg is None:
        return df

    for new_cols_type, cfg_new_columns in cfg.items():
        if new_cols_type == "single_cols":
            df = create_single_cols(df, cfg_new_columns)

        elif new_cols_type == "multiple_cols":
            df = create_multiple_cols(df, cfg_new_columns)

        else:
            logger.error(
                f"Invalid entry in `create_new_cols`. `{new_cols_type}` was given,"
                f"but only `single_cols` and `multiple_cols` are possible.",
            )
            sys.exit(1)

    return df


def create_single_columns(): ...


def create_multiple_columns(): ...


def load_dataset(cfg: InputFileConfig) -> pd.DataFrame:
    """Loads and preprocess dataset based on the dictionary.

    Args:
        cfg (InputFileConfig): a dictionary with a path, colum names, transformations, separator, etc.

    Returns:
        pd.DataFrame: a DataFrame made of read file.
    """
    df = read_raw_file(cfg.file_path, cfg.sheet_name)
    df = fix_column_names(df)
    df = filter_columns(df, cfg.keep_columns)
    df = create_new_columns(df, cfg.create_single_columns)
    df = create_new_columns(df, cfg.create_multiple_columns)

    logger.info(f"Reading data successful, a dataframe has a shape: {df.shape}.")

    return df


### MAIN


def is_string_integer(s: str) -> bool:
    """Checks if given string is an integer.

    Args:
        s (str): string to check.

    Returns:
        bool: True if given string is an integer, false otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_monocore_dict(df_dict: pd.DataFrame, value_col: str) -> dict:
    """Creates a dictionary out of the dataframe from a given column. It is then used to replace all rod values
    in input data for monocore values (like `keff`).

    Args:
        df_dict (pd.DataFrame): monocore dictionary.
        value_col (str): value to extract from a monocore dictionary (like `keff`).

    Returns:
        dict: dictionary that contains information about `rod_id` (or `rod_id + time_id`) and selected value.
    """
    index_id = df_dict["rod_id"].copy()
    if "time_id" in df_dict.columns:
        index_id += "_" + df_dict["time_id"]

    return df_dict.assign(index_id=index_id).set_index("index_id")[value_col].to_dict()


def split_column_names(col_names: list) -> tuple[list, list]:
    """Returns evolution column names and regular column names from a config settings.

    Returns:
        tuple[list, list]: a tuple that contains list of evolution columns and regular column respectively.
    """
    evolution_mask = np.char.endswith(col_names, "_evolution")

    evolution_cols = list(compress(col_names, evolution_mask))
    regular_cols = list(compress(col_names, ~evolution_mask))

    return evolution_cols, regular_cols


def filter_dict_by_suffix(input_dict: dict, suffix: str) -> dict:
    """Function filters dictionary by given suffix. It also removes that suffix and returns filtered dictionary
    without this suffix.

    Args:
        input_dict (dict): dictionary to filter.
        suffix (str): suffix to filter (and remove from dictionary).

    Returns:
        dict: filtered dictionary.
    """
    filtered_dict = {}
    for key in input_dict:
        rod_id, suffix_id = key.split("_")
        if suffix_id == suffix:
            filtered_dict[int(rod_id)] = input_dict[key]

    return filtered_dict


class CoreData:
    def __init__(self, cfg: ParcsannConfig) -> None:
        """Preprocess and stores core data. Divides them into `input` and `output` ready to feed to neural network.
        It also stores monocore data.

        Args:
            cfg (ParcsannConfig): a configuration file
        """
        self.cfg = cfg

        self.input_output_data = self.preprocess_input_output_data()
        self.monocore_data = self.preprocess_monocore_data()
        self.monocore_evolution_data = self.preprocess_monocore_evolution_data()

        self.input_data_raw = pd.DataFrame()
        self.input_data_df = pd.DataFrame()
        self.input_data_np = np.ndarray(0)

        self.output_data_df = pd.DataFrame()
        self.output_data_np = np.ndarray(0)

    # =================================================================================================================
    # PREPROCESS DATA
    # =================================================================================================================

    def preprocess_input_output_data(self) -> pd.DataFrame:
        input_output_data = load_dataset(self.cfg.input_output_file_details)
        input_output_data = input_output_data.rename(
            columns={col: f"pos{col}" for col in input_output_data.columns if is_string_integer(col)},
        )
        return input_output_data

    def preprocess_monocore_data(self) -> pd.DataFrame:
        """Reads monocore data and replaces column name `"`monocore` to `rod_id`.

        Returns:
            pd.DataFrame: monocore data.
        """
        monocore_data = load_dataset(self.cfg_data.MONOCORE_FILE_DETAILS)
        monocore_data = monocore_data.rename(columns={"monocore": "rod_id"})

        return monocore_data

    def preprocess_monocore_evolution_data(self) -> pd.DataFrame:
        """Creates dataframe with evolution of parameters (like keff) for monocores.

        Returns:
            pd.DataFrame: monocores evolution DataFrame.
        """
        monocore_evolution = pd.DataFrame()

        for rod_id in range(1, 10):
            logger.info(f"Reading and preprocessing data for {rod_id} monocore.")

            monocore_data = load_dataset(self.cfg_data.MONOCORE_EVOLUTION_FILE_DETAILS, sheet_name=f"{rod_id}")
            monocore_data["rod_id"] = f"{rod_id}"
            monocore_data["time_id"] = [f"t{i}" for i in range(1, len(monocore_data) + 1)]

            monocore_evolution = pd.concat((monocore_evolution, monocore_data))

        logger.info(f"Monocore evolution data successfully preprocessed: {monocore_evolution.shape}.")

        return monocore_evolution

    # =================================================================================================================
    # INPUT DATA PREPARE
    # =================================================================================================================

    def divide_core(self) -> Self:
        """Divides input data by given symmetry."""
        if self.cfg.CORE_SYMMETRY == "1/4":
            col_select_index = 32
        elif self.cfg.CORE_SYMMETRY == "1/8":
            col_select_index = 16
        else:
            logger.error(
                f"Core symmetry must be equal to 1/4 or 1/8, was given `{self.cfg.CORE_SYMMETRY}` instead.",
            )
            sys.exit(1)

        logger.info(
            f"Preparing input data, limiting core to the {self.cfg.CORE_SYMMETRY} symmetry "
            f"using {col_select_index} rod positions.",
        )
        self.input_data_raw = self.input_output_data.iloc[:, 0:col_select_index].copy()

        return self

    def transform_single_evolution_column(self, evolution_col: str, evolution_dict: dict) -> Self:
        """It replaces rod_id with corresponding values in evolution_dict for evolution_col. For each input value
        it creates T new columns, where T is the number of time steps. The result is added to the final input table.

        Args:
            evolution_col (str): name of the value that will replace `rod_id`.
            evolution_dict (): dictionary that contains evolution for each monocore.
        """
        time_steps = self.monocore_evolution_data["time_id"].unique()

        for time in time_steps:
            input_temp = self.input_data_raw.copy()
            input_temp = input_temp.add_suffix(f"_{evolution_col}_{time}")
            input_temp = input_temp.replace(filter_dict_by_suffix(evolution_dict, time))
            self.input_data_df = pd.concat((self.input_data_df, input_temp), axis=1)

        return self

    def transform_evolution_columns(self, evolution_cols: list) -> Self:
        """Transforms and adds to the final input table evolution for all given columns.

        Args:
            evolution_cols (str): list of evolution columns to add to the input table.
        """
        evolution_cols = [s.removesuffix("_evolution") for s in evolution_cols]

        for evolution_col in evolution_cols:
            logger.info(f"Adding `{evolution_col}` evolution to the input data.")
            evolution_dict = get_monocore_dict(self.monocore_evolution_data, evolution_col)
            self.transform_single_evolution_column(evolution_col, evolution_dict)

        return self

    def apply_single_one_hot_encoding(self) -> pd.DataFrame:
        """Applies one hot encoding, but instead of ones it leaves the original values.

        Returns:
            pd.DataFrame: a DataFrame with applied modified one hot encoding.
        """
        input_data_one_hot_encoded = pd.DataFrame()
        rod_ids = sorted(pd.melt(self.input_data_raw).value.unique())

        for rod_id in rod_ids:
            input_data_one_hot_encoded[[f"{col}_rod{rod_id}" for col in self.input_data_raw.columns]] = (
                self.input_data_raw == rod_id
            ) * rod_id

        return input_data_one_hot_encoded

    def transform_regular_columns(self, regular_cols: list) -> Self:
        """Replaces each `rod_id` with given monocore value. It may add more than one regular column. The data
        will be concatenated to the final input table.

        Args:
            regular_cols (list): list of values to add to the input table.

        """
        input_data_raw_temp = self.input_data_raw.copy()
        if self.cfg.ONE_HOT_ENCODING:
            logger.info("Applying one hot encoding to the input data.")
            input_data_raw_temp = self.apply_single_one_hot_encoding()

        for regular_col in regular_cols:
            logger.info(f"Adding `{regular_col}` to the input data.")
            regular_dict = get_monocore_dict(self.monocore_data, regular_col)
            input_data_temp = input_data_raw_temp.copy()
            input_data_temp = input_data_temp.add_suffix(f"_{regular_col}")
            input_data_temp = input_data_temp.replace(regular_dict)

            self.input_data_df = pd.concat((self.input_data_df, input_data_temp), axis=1)

    def apply_regular_one_hot_encoding(self) -> pd.DataFrame:
        """Applies one hot encoding.

        Returns:
            pd.DataFrame: input data with applied one hot encoding.
        """
        input_data_one_hot_encoded = self.input_data_raw.applymap(lambda rod_id: f"rod{rod_id}")
        input_data_one_hot_encoded = pd.get_dummies(input_data_one_hot_encoded, dtype=int, prefix_sep="_")

        return input_data_one_hot_encoded

    def prepare_input(self) -> Self:
        """Prepares input data, applies core symmetry, monocore dictionary and one hot encoding."""
        self.divide_core()

        if self.cfg.USE_MONOCORES:
            logger.info("Applying monocore information to the input data.")
            evolution_cols, regular_cols = split_column_names(self.cfg.INPUT_COLUMNS)

            if (len(evolution_cols) != 0) & self.cfg.ONE_HOT_ENCODING:
                if len(regular_cols) != 0:
                    logger.warning(
                        "Detected both evolution columns and one hot encoding. "
                        "One hot encoding cannot be applied to the evolution columns, only for regular one ",
                        f"specified in the configuration file: {regular_cols}.",
                    )
                else:
                    logger.warning(
                        "Detected both evolution columns and one hot encoding. "
                        "One hot encoding cannot be applied to the evolution columns, only for regular one, but ",
                        "there is no regular columns specified in the config. Proceed without one hot encoding.",
                    )

            self.transform_evolution_columns(evolution_cols)
            self.transform_regular_columns(regular_cols)

        else:
            self.input_data_df = self.input_data_raw.copy()
            if self.cfg.ONE_HOT_ENCODING:
                logger.info("Applying one hot encoder.")
                self.input_data_df = self.apply_regular_one_hot_encoding()

        self.input_data_np = self.input_data_df.to_numpy().copy()
        return self

    # =================================================================================================================
    # OUTPUT DATA PREPARE
    # =================================================================================================================

    def find_evolution_cols(self, evolution_col: str) -> list:
        """Finds and returns all columns from the input-output data that contains `evolution_col` followed by a number.
        But first it removes suffix `_evolution` from a `evolution_col`.

        Args:
            evolution_col (str): column with a `_evolution` suffix to find in the dataset.

        Returns:
            list: list of all columns from input-output data that matches `evolution_col` patter followed by a number.
        """
        evolution_col = evolution_col.replace("_evolution", "")
        pattern = rf"^{evolution_col}.*\d+$"

        evolution_cols = [col for col in self.input_output_data.columns.to_numpy() if re.match(pattern, col)]

        if not evolution_cols:
            logger.error(
                f"Did not found any column that matches the pattern `{pattern}`. Please check if it is a correct name.",
            )
            sys.exit(1)

        logger.info(f"Found {len(evolution_cols)} columns in the data that matches the pattern `{pattern}`.")

        return evolution_cols

    def get_output_col_names(self) -> list:
        """Returns a list of the names of all output columns.

        Returns:
            list: list of names of all output columns.
        """
        evolution_cols, output_cols = split_column_names(self.cfg.OUTPUT_COLUMNS)

        if not evolution_cols:
            return output_cols

        for evolution_col in evolution_cols:
            output_cols += self.find_evolution_cols(evolution_col)

        return output_cols

    def prepare_output(self) -> Self:
        """Filter output data based on the output columns from a config file."""
        logger.info("Preparing output data.")
        self.output_data_df = self.input_output_data.loc[:, self.get_output_col_names()].copy()
        self.output_data_np = self.output_data_df.to_numpy().copy()
        logger.info(
            f"Prepared output data, keeping {self.output_data_df.shape[1]} columns: {self.cfg.OUTPUT_COLUMNS}.",
        )

        return self

    # =================================================================================================================
    # MODELING PART
    # =================================================================================================================

    def train_test_div(self, test_size: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train as a `test_size` fraction."""
        if test_size is None:
            test_size = self.cfg.TRAIN_SPLIT

        return train_test_split(self.input_data_np, self.output_data_np, test_size=test_size)

    # =================================================================================================================
    # CALLABLES
    # =================================================================================================================

    def __call__(self, cfg: ConfigModeling | None = None) -> Self:
        if (self.cfg is None) & (cfg is None):
            logger.error("The Modeling Config was not provided. Add it in the init or call function.")
            sys.exit(1)

        if (self.cfg is not None) & (cfg is not None):
            logger.warning(
                "The Modeling Config was provided both in the init and in the call function. "
                "The one from the call function will be used.",
            )

        self.cfg = cfg

        self.prepare_input()
        self.prepare_output()

        return self
