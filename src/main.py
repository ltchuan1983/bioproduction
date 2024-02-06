import argparse
import json
import yaml
from joblib import dump, load
import sys

import pandas as pd
import numpy as np

from helper import parse_args, load_sql_data, load_split_save_sql_data, load_split_XY, check_tables_in_db, remove_data_by_mw, r2_rmse_score
from pipelines import create_pipe, create_preprocessor
from modes import perform_cross_validation, perform_train_test, run_train, run_train_multi, run_train_gridsearch, run_train_bayes, run_train_nn

from sklearn.metrics import r2_score, root_mean_squared_error


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

TARGETS = ['yield', 'titer', 'rate']
H2_MW = 2

NUM_FOLDS = 8


def main():
    """
    Options:
    1. Run predefined regressor both CV and train_test ways (e.g. CatBoostRegressor). Ask for which db to use. Ask to save model or not
    2. Run predefined list of regressor. Ask for which db to use. Ask to save model or not.
    3. Run hyperparameter tuning for predefined regressor
    4. Run neural network
    5. Run hyperparameter tuning for neural network
    6. Run AutoML
    7. Run stacking
    8. Use saved model to predict 

    
    """

    # Load train and test data from sqlite db. Still needs further preprocessing.
    X_train, y_train = load_split_XY("train_data")
    X_test, y_test = load_split_XY("test_data")

    # Preprocess X_train and X_test
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    if MODE == "train":
        run_train(X_train, y_train, X_test, y_test)

    if MODE == "train_multi":
        run_train_multi(X_train, y_train, X_test, y_test)

    if MODE == "train_gridsearch":
        run_train_gridsearch(X_train, y_train, X_test, y_test)
    
    if MODE == "train_bayes":
        run_train_bayes(X_train, y_train, X_test, y_test)
    
    if MODE == "train_nn":
        run_train_nn(X_train, y_train, X_test, y_test)


if __name__ == "__main__":

    # Prepare and save train and test 
    load_split_save_sql_data(test_size=0.3)
    check_tables_in_db()


    # df_train = remove_data_by_mw(df_train, 2)
    # df_test = remove_data_by_mw(df_test, 2)

    args = parse_args()

    MODE = args.mode

    main()

   
