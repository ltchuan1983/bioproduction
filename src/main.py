"""
Main module

This module runs the selected end-to-end pipeline given the option entered at command line.

Function:
    main: Run pipeline when the module is executed as the main program, depending on the argument at command line
"""

import pandas as pd

from helper import parse_args, remove_data_by_mw, r2_rmse_score
from pipelines import load_sql_data, load_split_save_sql_data, load_split_XY, check_tables_in_db, load_split_preprocess, load_and_augment_targets, load_and_augment_features, load_test_data
from modes import run_train, run_train_multi, run_train_gridsearch, run_train_bayes, run_train_nn, run_train_embed_nn, run_train_embed_genotype_nn, run_train_tunable_nn, run_train_automl, run_train_stack, run_train_stack_nn2embed_catboost, run_train_stack_nn1embed_catboost, run_predict
from helper import convert_to_num_lists, convert_str_to_numpy_array, convert_nocomma_str_to_list

from sklearn.metrics import r2_score, root_mean_squared_error


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

def main():
    """ Main function to run end-to-end pipeline of reading sqlite3 data, preprocessing data, fitting model and making predictions

    Options:
    1. Run CatBoostRegressor in both CV and train_test ways. Ask for which db to use. Ask to save model or not
        cleaned_table   ---> default, excludes information on strain_background_genotype
        cleaned_table_2 ---> Includes embedding vectors (aggregated via averaging) on strain_background_genotype
        cleaned_table_3 ---> Includes embedding vectors (aggregated via appending) on strain_background_genotype
    2. Run CatBoostRegressor in both CV and train_test ways for augmented data.
        Can also use among cleaned_table, cleaned_table_2 or cleaned_table_3
    3. Run predefined list of regressor. Ask for which db to use. Ask to save model or not.
    4. Run GridSearchCV hyperparameter tuning for CatBoostRegressor
    5. Run BayesSearchCV hyperparameter tuning for CatBoostRegressor
    6. Run neural network
    7. Run neural network (Onehot_encoded categorical features fed into Embedding Layer)
    8. Run neural network (2 separate embedding layers for onehot encoded categorical features and strain_background_genotype)   
    9. Run hyperparameter tuning for neural network using Keras Tuner
    10. Run H2O AutoML
    11. Run stacking (CatBoostRegressor -> LinearRegression)
    12. Run stacking (neural network -> CatBoostRegressor)
        Use augmented data as well as 1 embedding layer for onehot encoded categorical features
    13. Run stacking (neural network -> CatBoostRegressor)
        Use augmented data as well as 2 separate embedding layers for onehot encoded categorical features and strain_background_genotype
    14. Use saved model to predict

    """

    # 1
    if MODE == "train":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train(X_train, y_train, X_test, y_test)

    #2
    if MODE == "train_augmentdata":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        # Same as mode=="train" when augmentation = 0
        X_train, y_train, X_test, y_test = load_and_augment_targets("cleaned_data")
        run_train(X_train, y_train, X_test, y_test)

    #3
    if MODE == "train_multi":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_multi(X_train, y_train, X_test, y_test)

    #4
    if MODE == "train_gridsearch":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_gridsearch(X_train, y_train, X_test, y_test)

    #5
    if MODE == "train_bayes":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_bayes(X_train, y_train, X_test, y_test)
    
    #6
    if MODE == "train_nn":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_nn(X_train, y_train, X_test, y_test)

    #7
    if MODE == "train_embed_nn":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_embed_nn(X_train, y_train, X_test, y_test)

    #8
    if MODE == "train_embed_genotype_nn":
        # Data source msut be cleaned_data_4
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data_4')
        convert_to_num_lists(X_train, ["strain_background_genotype_tokenized"])
        convert_to_num_lists(X_test, ["strain_background_genotype_tokenized"])
        run_train_embed_genotype_nn(X_train, y_train, X_test, y_test)
    
    #9
    if MODE == "train_tunable_nn":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_tunable_nn(X_train, y_train, X_test, y_test)

    #10
    if MODE == "train_automl":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        run_train_automl(X_train, y_train, X_test, y_test)
    
    #11
    if MODE == "train_stack":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        X_train, y_train, X_test, y_test = load_and_augment_targets('cleaned_data')
        run_train_stack(X_train, y_train, X_test, y_test)

    #12
    if MODE == "train_stack_nn1embed_catboost":
        # Data source can be cleaned_data, cleaned_data_2, cleaned_data_3
        # X_train, y_train, X_test, y_test = load_split_preprocess('cleaned_data')
        X_train, y_train, X_test, y_test = load_and_augment_features("cleaned_data")
        run_train_stack_nn1embed_catboost(X_train, y_train, X_test, y_test)

    #13
    if MODE == "train_stack_nn2embed_catboost":
        # Data source must be cleaned_data_4
        X_train, y_train, X_test, y_test = load_and_augment_features("cleaned_data_4")
        convert_to_num_lists(X_train, ["strain_background_genotype_tokenized"])
        convert_to_num_lists(X_test, ["strain_background_genotype_tokenized"])
        run_train_stack_nn2embed_catboost(X_train, y_train, X_test, y_test)

    # 14
    # Need to specify optional argument <table_name> to read test data from 
    if MODE == "predict":
        X_test, y_test = load_test_data(args.table)
        convert_to_num_lists(X_test, ["strain_background_genotype_tokenized"])
        print(type(X_test['strain_background_genotype_tokenized'][438]))
        run_predict(X_test, y_test)
    

if __name__ == "__main__":

    args = parse_args()

    MODE = args.mode

    main()

   
