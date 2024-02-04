import argparse
import json
import yaml
from joblib import dump, load
import sys

import pandas as pd
import numpy as np

from helper import load_sql_data, load_split_save_sql_data, check_tables_in_db, remove_data_by_mw, r2_rmse_score
from pipelines import create_pipe, create_preprocessor
from modes import perform_cross_validation, perform_train_test

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error

from catboost import CatBoostRegressor

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

    if MODE == "train":
        run_train()


def run_train():

    # Perform cross validation
    regressor = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)
    perform_cross_validation(X_train.copy(), y_train, regressor)

    # Perform training, then validation on test data
    perform_train_test(X_train, y_train, X_test, y_test, regressor, "CatBoostRegressor")


def parse_args():

    # Create ArgmentParser instance
    parser = argparse.ArgumentParser(
        prog = "training_and_prediction_options",
        description = "Options to use different pipelines for model training and to make predictions"
    )

    # Add positional argument
    parser.add_argument('mode', type=str, help='pipe->Single regressor')

    # Parse the command line arguments

    try:
        args = parser.parse_args()
        
    except SystemExit:
        print("Options for mode:")
        print("------------------")
        print("train -----> Fit CatBoostRegressor on train data and perform predictions on test data")
        sys.exit(1)
  
    return args

if __name__ == "__main__":

    # Prepare and save train and test 
    load_split_save_sql_data(test_size=0.3)
    check_tables_in_db()

    # Load train and test data from sqlite db. Still needs further preprocessing.
    df_train = load_sql_data("train_data")
    df_test = load_sql_data("test_data")

    X_train = df_train.drop(columns=TARGETS, axis=1)
    y_train = df_train[TARGETS]

    X_test = df_test.drop(columns=TARGETS, axis=1)
    y_test = df_test[TARGETS]

    # df_train = remove_data_by_mw(df_train, 2)
    # df_test = remove_data_by_mw(df_test, 2)

    args = parse_args()

    MODE = args.mode

    main()

   
