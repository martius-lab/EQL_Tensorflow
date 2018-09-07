"""
Aggregation and scoring of multiple results.
    - *aggregate_csv_files_recursively* collects the results in a given directory in a pandas dataframe.
    - *select_instance* selects the best performing model instance. It expects a pandas dataframe or a filename of a
    file containing the results of each model instance and selects the best instance based on validation-
    and extrapolation-performance or validation-performance and complexity, depending on the availability of extrapolation data.
    - Running *model_selection.py* executes both of these routines.
"""
from ast import literal_eval
from os import path, walk
from sys import argv

import numpy as np
import pandas as pd


def select_instance(df=None, file=None, use_extrapolation=True):
    """
    Expects a file with one row per network and columns reporting the parameters and complexity and performance
    First line should be the column names, col1 col2 col3..., then one additional comments line which can be empty.
    Third line should be the values for each column.
    :param df: pandas dataframe containing data about model performance
    :param file: file containing data about model performance, only used if dataframe is none
    :param use_extrapolation: flag to determine if extrapolation data should be used
    :return: pandas dataframe containing id and performance data of best model.
    """
    if df is not None and file is not None:
        raise ValueError('Both results_df and file specified. Only specify one.')
    if df is None:
        if file is None:
            raise ValueError('Either results_df or file have to be specified.')
        df = pd.read_csv(file)
    if 'extr_error' in df.keys():
        extr_available = not df['extr_error'].isnull().values.any()
    else:
        extr_available = False
    if use_extrapolation and not extr_available:
        raise ValueError("use_extrapolation flag is set to True but no extrapolation results were found.")

    if use_extrapolation:
        df['extr_normed'] = normalize_to_zero_one(df['extr_error'])
    df['val_normed'] = normalize_to_zero_one(df['val_error'])
    df['complexity_normed'] = normalize_to_zero_one(df['complexity'], defensive=False)

    if use_extrapolation:
        print('Extrapolation data used.')
        df['score'] = np.sqrt(df['extr_normed'] ** 2 + df['val_normed'] ** 2)
    else:
        print('No extrapolation data used, performing model selection based on complexity and validation instead.')
        df['score'] = np.sqrt(df['complexity_normed'] ** 2 + df['val_normed'] ** 2)

    scored_df = df.sort_values(['score'])
    best_instance = scored_df.iloc[[0]]
    return best_instance, scored_df


def normalize_to_zero_one(arr, defensive=True):
    """
    Routine that normalizes an array to zero and one.
    :param arr: array to be normalized
    :param defensive: flag to determine if behavior is defensive (if all array elements are the same raise exception)
                      or not (if all array elements are the same return an array of same length filled with zeros)
    """
    if np.isclose(np.max(arr), np.min(arr)):
        if defensive:
            raise ValueError('All elements in array are the same, no normalization possible.')
        else:
            return np.zeros(len(arr))
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr


def aggregate_csv_files_recursively(directory, filename):
    """ Returns a pandas DF that is a concatenation of csvs with given filename in given directory (recursively)."""
    return pd.concat(_df_from_csv_recursive_generator(directory, filename))


def _df_from_csv_recursive_generator(directory, filename):
    """ Returns a generator producing pandas DF for each csv with given filename in given directory (recursively)."""
    for root, dirs, files in walk(directory):
        if filename in files:
            yield pd.read_csv(path.join(root, filename))


if __name__ == '__main__':
    if len(argv) > 1:
        passed_dict = literal_eval(argv[1])
        results_path = passed_dict['results_path']
        use_extrapolation = passed_dict['use_extrapolation']
    else:
        raise ValueError('Path to results directory must be passed.')
    aggregated_results = aggregate_csv_files_recursively(results_path, "results.csv")
    best_instance, ordered_instances = select_instance(df=aggregated_results, use_extrapolation=use_extrapolation)
    ordered_instances.to_csv(path.join(results_path, "scored_results.csv"))
    print('All instances, ordered by score:\n', ordered_instances)
    print('Selected model instance:\n', best_instance)
