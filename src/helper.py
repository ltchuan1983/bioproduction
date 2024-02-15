"""
    Helper module
    
    This module provides helper functions for the following purposes:
    1. Parsing command line arguments
    2. Functions to build steps in preprocessing pipelines
    3. Functions to compute and display evaluation metrics
    4. Functions to create keras neural network models
"""

import sys
import argparse
import random
import pandas as pd
import numpy as np

from scipy.stats import boxcox

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline

import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten, Concatenate, Embedding
from keras.optimizers import Adam
from keras import regularizers

from catboost import CatBoostRegressor

from config import Config

config = Config()

SEED = config.config_data['SEED']

ONEHOT_EMBED_DIM = config.config_data['ONEHOT_EMBED_DIM']
GENOTYPE_EMBED_DIM = config.config_data['GENOTYPE_EMBED_DIM']

TOKENIZED_GENOTYPE_LEN = config.config_data['TOKENIZED_GENOTYPE_LEN']
REGULARIZATION_STRENGTH = config.config_data['REGULARIZATION_STRENGTH']
LEARNING_RATE = config.config_data['LEARNING_RATE']

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


def parse_args():
    """Create ArgumentParser, set positional arguments, parse and return command line arguments

    Returns:
        args (dict): Dictionary pairing "MODE" to choice passed in at command line
    """
    # Create ArgmentParser instance
    parser = argparse.ArgumentParser(
        prog = "training_and_prediction_options",
        description = "Options to use different pipelines for model training and to make predictions"
    )

    mode_choices = ['train', 'train_multi', 'train_gridsearch', 'train_bayes', 'train_nn', 'train_embed_nn', 'train_embed_genotype_nn', 'train_tunable_nn', 'train_automl', 'train_stack', 'train_stack_nn1embed_catboost', 'train_stack_nn2embed_catboost', 'predict']

    # Add positional argument
    parser.add_argument('mode', type=str, choices=mode_choices, metavar='mode', help=f'Mode of operation. Choose one of the following {mode_choices}')

    # Add optional argument
    parser.add_argument('--table', type=str)

    # subparsers = parser.add_subparsers(dest="mode")
    # predict_parser = subparsers.add_parser('predict')
    # predict_parser.add_argument('--table', type=str, help="Name of table containing data used to generate predictions")



    # Parse the command line arguments
    try:
        args = parser.parse_args()
        
    except SystemExit:
        print("Check error message for permissible choices")
        sys.exit(1)
  
    return args

####################################################
#### Functions for use in preprocessing pipeline
####################################################


def create_gene_list(df, gene_string_features):
    """ 
    Convert one single string into list of strings where each string is a gene

    A function to call by passing df and features 
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed

    Returns:
        df (pd DataFrame): Data where strings have been converted to list of strings
    """
    for feature in gene_string_features:
        #new_feature = gene_list_feature + "_num"
        for index, row in df.iterrows():
            gene_list = row[feature].split(', ')
            df.at[index, feature] = gene_list
    return df

def count_genes_per_row(list):
    """
    Count the number of genes (one string per gene) in the list
    If the only string is 'nil', return 0

    A function to be passed into df.apply()
    NOT a function to call directly

    Args:
        list (list): list of strings
    
    Returns:
        length (int): length of list or 0
    """
    length = len(list)
    if length > 1:
        return length
    if length == 1:
        if list[0] == 'nil':
            return 0
        else:
            return 1

def count_genes(df, gene_string_features):
    """
    Counts the number of genes (i.e. strings) in each row of DataFrame.
    Use function count_genes_per_row 

    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed
    
    Returns: 
        df (pd DataFrame): Data with new feature containing total num of genes
    """
    for feature in gene_string_features:
        tally_name = feature + '_num'
        df[tally_name] = df[feature].apply(count_genes_per_row)
    
    return df

def convert_to_num_list_per_row(string_list):
    """ 
    Function to convert string version of a num list to a list of numbers and handle errors if any

    A function to be passed into df.apply()
    NOT a function to call directly

    Args:
        string_list (str): string version of num list in each cell to be converted back into actual num list
    
    Returns:
        list (list): list of numbers or None 
    """
    try:
        return eval(string_list)
    except (SyntaxError, ValueError):
        return None

def convert_to_num_lists(df, gene_numlist_features):
    """
    Function to convert strings in each row into lists of numbers
    Use function convert_to_num_list_per_row

    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed
        gene_numlist_features (list of str): Features to convert strings into lists of numbers
    
    Returns:
        df (pd DataFrame): Data after preprocessing
    """
    for feature in gene_numlist_features:
        df[feature] = df[feature].apply(convert_to_num_list_per_row)
    
    return df

def apply_log1p(df, num_features):
    """
    Function to perform log1p transformation on select numerical features

    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed
        num_features: Features to be transformed
    
    Returns:
        df (pd DataFrame): Data after preprocessing

    """
    for feature in num_features:
        df[feature] = df[feature].apply(np.log1p)
    
    return df

def convert_str_to_numpy_array(df, feature):
    """
    Function to convert strings into numpy arrays on a single select feature

    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed
        feature (str): Feature to be transformed
    
    Returns:
        df (pd DataFrame): Data after preprocessing

    """

    for i, row in df.iterrows():
        df.at[i, feature] = np.fromstring(row[feature].strip('[]'), dtype=np.float32, sep=' ')
    
    return df

def convert_nocomma_str_to_list(df, feature):
    """
    Function to convert a list-like, comma-less string into a list on a single select feature

    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed
        feature (str): Feature to be transformed
    
    Returns:
        df: Data after preprocessing

    """
    
    for i, row in df.iterrows():
        array_str = row[feature]
        elements = array_str.strip('[]').split(" ")
        #integer_list = [int(element) for element in elements]
        df.at[i, feature] = elements

    return df

def tally_num_lists(df, gene_numlist_features):
    """
    Function to sum up all the numbers in the list per row for select features in the df

    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): Data to be preprocessed
        gene_numlist_features (list of str): Features to tally
    
    Returns:
        df: Data after preprocessing
    """
    for feature in gene_numlist_features:
        tally_name = feature + '_num'
        df[tally_name] = df[feature].apply(sum)
    
    return df

def onehot_encode(df, categorical_features):
    """
    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 
    """

    for feature in categorical_features:
        df = pd.get_dummies(df, columns=[feature], prefix=feature, dtype=int)
    
    return df

def remove_features(df, features_to_drop):
    """
    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 
    """

    df = df.drop(columns=features_to_drop, axis=1)
    return df

def remove_data_by_mw(df, mw):
    """
    Remove data rows where mw matches the value of mw passed in.

    A function to call by passing df and features 
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Args:
        df (pd DataFrame): DataFrame containing data to be processed
        mw (int): Molecular weight of product (e.g. 2 for hydrogen) 

    Returns:
        df (pd DataFrame): DataFrame where rows specified by mw have been removed
    """

    df = df.drop(df[df.mw==mw].index, axis=0)

    return df

def log1p_transformer(X):
    """
    Perform log1p transformation on array

    Args:
        X (array): data in array

    Returns:
        Array after log1p transformation
    """
    return np.log1p(X)

def boxcox_transformer(df):
    """
    
    Perform box-cox transformation on DataFrame passed in

    Args:
        df (DataFrame): Data

    Returns:
        df (DataFrame): Data after boxcox transformation
    """
    for feature in df.columns:  # Transpose to iterate over features
        transformed_column, _ = boxcox(df[feature]+1)
        df[feature] = transformed_column
    return df

def scale_X_and_Y(X_train, y_train, X_test, y_test, noscale_features):
    """
    Performs log1p transformations and Standard Scaling on numerical features
    Performs Standard Scaling on target values

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
        noscale_features (list of str): Features that should not be transformed and scaled

    Returns:
        X_train_scaled (DataFrame or array): Scaled Features in train data
        y_train_scaled (DataFrame or array): Scaled Targets in train data
        X_test_scaled (DataFrame or array): Scaled Features in test data
        y_test_scaled (DataFrame or array): Scaled Targets in test data
        all_columns (list or str): Column headings, numerical features then unscaled features
        ScalerY (object): Pass scaler back to inverse transform the predictions
    """

    numerical_features = [col for col in X_train.columns if col not in noscale_features]

    Log1P = FunctionTransformer(apply_log1p, kw_args={'num_features': numerical_features})
    
    num_pipe = Pipeline([
        ('log1p', FunctionTransformer(func=log1p_transformer)),
        ('scaler', StandardScaler())
    ])

    ## Cannot add a second step like this. Will add another set of numerical_features
    ## Best to consolidate all steps in a pipeline and run the pipeline once inside ColumnTransformer
    # column_transformer = ColumnTransformer(
    #     transformers=[
    #         ('scaler', StandardScaler(), numerical_features),
    #         ('log1p', FunctionTransformer(func=log1p_transformer), numerical_features)
    #     ], 
    #     remainder='passthrough' # passthrough columns will be placed at the back of the df
    # )

    column_transformer = ColumnTransformer(
        transformers=[
            ('num_pipe', num_pipe, numerical_features)
        ], 
        remainder='passthrough' # passthrough columns will be placed at the back of the df
    )

    # Change the order of columns so that passthrough columns are at the back
    all_columns = numerical_features + noscale_features
    X_train_reordered = X_train.reindex(columns=all_columns)
    X_test_reordered = X_test.reindex(columns=all_columns)

    X_train_scaled = column_transformer.fit_transform(X_train_reordered)
    X_test_scaled = column_transformer.transform(X_test_reordered)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=all_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=all_columns)


    scalerY = StandardScaler()
    y_train_scaled = scalerY.fit_transform(y_train)
    y_test_scaled = scalerY.transform(y_test)

    y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, scalerY, column_transformer



#########################################################
#### Functions to compute and display evaluation metrics
#########################################################


def r2_rmse_score(y_test, y_pred, model_name):

    """
        Function to compute and display r2_square

        Args:
            y_test (pd DataFrame or array): Target values
            y_pred (pd DataFrame or array): Model predictions
            model_name (str): Name of model being evaluated
        
        Returns:
            r2_val (float): Output from r2_score
    """

    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])
    y_test = pd.DataFrame(y_test, columns=['yield', 'titer', 'rate'])

    r2_val = r2_score(y_test, y_pred)

    print(" ")
    print("*"*50)
    print(f'Train Test Split Metrics for {model_name}')
    print("*"*50)
    print(" ")
    print('R_squared score for overall: ', r2_val)
    print('R_squared score for yield: ', r2_score(y_test['yield'], y_pred['yield']))
    print('R_squared score for titer: ', r2_score(y_test['titer'], y_pred['titer']))
    print('R_squared score for rate: ', r2_score(y_test['rate'], y_pred['rate']))
    print(" ")
    print('RMSE for overall: ', root_mean_squared_error(y_test, y_pred))
    print('RMSE for yield: ', root_mean_squared_error(y_test['yield'], y_pred['yield']))
    print('RMSE for titer: ', root_mean_squared_error(y_test['titer'], y_pred['titer']))
    print('RMSE for rate: ', root_mean_squared_error(y_test['rate'], y_pred['rate']))

    return r2_val

def display_cv_score(score, type):
    """
    Function to display scores from cross_val_score

    Args:
        score (array): array of scores from various folds
        type (str): Name describing type of score
    """

    print("**************************************")
    print(f"Cross Validation Metrics: {type}")
    print("**************************************")
    print(score)
    print("Average: ", score.mean())

def rmse(y_true, y_pred):
    """
    Function to compute rmse

    Args:
        y_true (DataFrame or array): Target values
        y_pred (DataFrame or array): Model Predictions

    Returns:
        float: rmse score
    """
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

@keras.saving.register_keras_serializable(package="src", name="r_squared")
def r_squared(y_true, y_pred):
    """
    Function to return Tensor operation to compute r2_square as a metric in Keras model

    Args:
        y_true (DataFrame or array): Target values
        y_pred (DataFrame or array): Model Predictions

    Returns:
        tf.py_function() (Tensor operation): Operation to compute r2_squared as a metric in Keras Model
    """


    ## Alternative definition of r2_squared
    ## BUT using r2_score to be consistent
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

    # Custom defined metric by using tf.py_function as wrapper
    # return tf.py_function(r2_score, (y_true, y_pred), tf.float64)

def perform_cross_validation(X_train, y_train, regressor):
    """
    Function to run cross_val_score (once for r2 and once for neg rmse) and display scores

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        regressor (object): Regressor model instance
    """

    NUM_FOLDS = config.config_data['NUM_FOLDS']

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=40)

    cross_val_r2 = cross_val_score(regressor, X_train, y_train, cv=kf, scoring='r2')
    cross_val_negrmse = cross_val_score(regressor, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

    display_cv_score(cross_val_r2, "R-squared")
    display_cv_score(-cross_val_negrmse, "RMSE")

def perform_train_test(X_train, y_train, X_test, y_test, regressor, regressor_name):
    """
    Function to fit regressor on train data and make predict on test data

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
        regressor (object): Regressor model instance
        regressor_name (str): Name of regressor to be displayed with scores
    
    Returns:
        y_pred_reduced (DataFrame): Predictions from reduced set of features
    """


    
    regressor.fit(X_train, y_train) # should save model

    y_pred = regressor.predict(X_test)

    r2_rmse_score(y_test, y_pred, regressor_name)

    return regressor

def extract_features_to_filter(feature_columns, regressor, DISPLAY_IMPORTANCES=True):
    """Extract feature importances from the regressor and specify features with low importance (below a threshold)

    Args:
        feature_columns (list): List of column headings for features
        regressor (object): Regressor that has been fitted
        DISPLAY_IMPORTANCES (bool, optional): Display the importances of each feature. Defaults to True.

    Returns:
        [type]: [description]
    """

    IMPORTANCE_THRESHOLD = config.config_data['IMPORTANCE_THRESHOLD']
    
    feature_importances = zip(feature_columns, regressor.feature_importances_)

    sorted_feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)

    if DISPLAY_IMPORTANCES:
        for item in sorted_feature_importances:
            print(item[0], '---->', item[1])
    
    features_to_filter = [feature for feature, importance in sorted_feature_importances if importance < IMPORTANCE_THRESHOLD]

    return features_to_filter

def perform_train_test_reduced_features(X_train, y_train, X_test, y_test, features_to_filter, regressor, regressor_name):
    """
    Function to fit regressor on train data with reduced feature set and 
    make predict on test data (also with reduced feature set)
    
    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
        features_to_filter: List of features to remove due to low importances
        regressor (object): Regressor model instance
        regressor_name (str): Name of regressor to be displayed with scores
    
    Returns:
        y_pred_reduced (DataFrame): Predictions from reduced set of features
        regressor (Object): Regressor fitted on train data with reduced feature set
    """
    X_train_reduced = X_train.drop(features_to_filter, axis=1)
    X_test_reduced = X_test.drop(features_to_filter, axis=1)

    regressor.fit(X_train_reduced, y_train)

    y_pred_reduced = regressor.predict(X_test_reduced)

    # y_pred output from pipe is numpy.array, so need to convert
    y_pred_reduced = pd.DataFrame(y_pred_reduced, columns=['yield', 'titer', 'rate'])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test, y_pred_reduced, regressor_name + ' filter features')
    return y_pred_reduced, regressor

##############################################
#### Functions to create neural network models
##############################################


def make_nn_model_1output():
    """ Create keras model to predict 1 target 
    """

    input_dim = 42
    output_dim = 1

     # tanh performs much better than ReLU and LeakyReLU
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(2048, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(64, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(Dense(output_dim, activation="tanh"))

    # Compile the model
    # r_squared uses r2_score from sklearn.preprocessing, not custom defiend
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=[r_squared])

    return model

def make_nn_model_3outputs():
    """ Create keras model to predict 3 targets concurrently
    """

    input_dim = 42
    output_dim = 3

     # tanh performs much better than ReLU and LeakyReLU
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(2048, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(64, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(Dense(output_dim, activation="tanh"))

    # Compile the model
    # r_squared uses r2_score from sklearn.preprocessing, not custom defiend
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=[r_squared])

    return model

def make_tunable_nn_model(hp):
    """ Function to create tunable neural network model for use with keras_tuner

    Example:
    tuner = keras_tuner.BayesianOptimization(
            make_tunable_nn_model,
            seed=808,
            objective='val_loss',
            max_trials=15,
            directory='../models',
            project_name='keras_tuner_Bayes'
    )

    Args:
        hp (Object): HyperParameters instance from Keras Tuner Library

    Returns:
        model: keras neural network model
    """

    input_dim = 42
    output_dim = 3

    SEED = config.config_data['SEED']

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = Sequential()

    # Input layer
    model.add(Dense(units=hp.Int('input_dense', min_value=64, max_value=512, step=16), input_dim=input_dim, kernel_initializer='glorot_uniform', activation='tanh'))

    # Tunable number of blocks containing 1 Dropout, 1 BatchNormalization and 1 Dense layer
    # for i in range(hp.Int('norm_dropout_dense', min_value=1, max_value=5, step=1)):
    #     model.add(BatchNormalization())
    #     model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.05)))
    #     model.add(Dense(units=hp.Int('hidden_dense', min_value=64, max_value=4096, step=64), kernel_initializer='glorot_uniform', activation='tanh'))

    # 3-block architecture that works well over various trials
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(units=hp.Int('hidden_dense_1', min_value=64, max_value=4096, step=16), kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(units=hp.Int('hidden_dense_2', min_value=64, max_value=4096, step=16), kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(units=hp.Int('hidden_dense_3', min_value=64, max_value=4096, step=16), kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(Dense(output_dim, activation="tanh"))

    # Output layer
    model.add(Dense(output_dim, activation="tanh"))

    # # Setting up choices for different optimizers
    # hp_optimizer = hp.Choice("Optimizer", values=['Adam', 'SGD'])

    # if hp_optimizer == 'Adam':
    #     hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    # elif hp_optimizer == 'SGD':
    #     hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
    #     nesterov = True
    #     momentum = 0.9

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=[r_squared])
    
    return model

def make_nn_model_embed_cat(num_features_count, onehot_features_count):
    """
    Function to create keras neural network model with one embedding layer for the one-hot encoded
    categorical features

    Args:
        num_features_count (int): Number of numerical features e.g. cs_conc1
        onehot_features_count (int): Number of onehot_features i.e. categorical features

    Returns:
        model: Keras neural network model
    """

    # Define input layers
    num_input = Input(shape=(num_features_count, ), name='num_input')
    onehot_input = Input(shape=(onehot_features_count, ), name='onehot_input')

    # Define embedding layer for one-hot features
    # First step will produce output with shape (batch size, input_length, output_dim) (each dense vector represents each column)
    embedding_output = Embedding(input_dim=onehot_features_count, output_dim=ONEHOT_EMBED_DIM, input_length=onehot_features_count)(onehot_input)
    # Flatten into (batch size, batch size x input_length)
    embedding_output = Flatten()(embedding_output)

    # Layer to concatenate continous_input and embedding output
    concatenated = Concatenate()([num_input, embedding_output])

    # Additional blocks of Dense, Dropout and BatchNormalization Layers
    concatenated = Dense(64, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.15)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(1024, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.15)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(256, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.15)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(32, kernel_initializer='glorot_normal', activation='tanh')(concatenated)

    # Output layer
    output = Dense(3, activation='linear')(concatenated)

    # Deifine model
    model = Model(inputs=[num_input, onehot_input], outputs=output)

    # Compile the model
    optimizer =  Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r_squared])

    return model

def make_nn_model_embed_cat_genotype(num_features_count, onehot_features_count):

    """
    Function to create keras neural network model with one embedding layer for the one-hot encoded
    categorical features and another embedding layer for tokenized strain_background_genotype

    Args:
        num_features_count (int): Number of numerical features e.g. cs_conc1
        onehot_features_count (int): Number of onehot_features i.e. categorical features

    Returns:
        model: Keras neural network model
    """

    # Define input layers
    num_input = Input(shape=(num_features_count, ), name='num_input')
    onehot_input = Input(shape=(onehot_features_count, ), name='onehot_input')
    # tokenized genotypes padded to length of 20
    genotype_input = Input(shape=(TOKENIZED_GENOTYPE_LEN, ), name="genotype_input")

    # Define embedding layer for strain_background_genotype_tokenized
    embedding_output_genotype = Embedding(input_dim=199, output_dim=GENOTYPE_EMBED_DIM, input_length=20, embeddings_regularizer=regularizers.l2(REGULARIZATION_STRENGTH))(genotype_input)
    embedding_output_genotype = Flatten()(embedding_output_genotype)

    # Define embedding layer for one-hot features
    # First step will produce output with shape (batch size, input_length, output_dim) (each dense vector represents each column)
    embedding_output_onehot = Embedding(input_dim=onehot_features_count, output_dim=ONEHOT_EMBED_DIM, input_length=onehot_features_count)(onehot_input)
    # Flatten into (batch size, batch size x input_length)
    embedding_output_onehot = Flatten()(embedding_output_onehot)

    # Layer to concatenate continous_input and embedding output
    concatenated = Concatenate()([num_input, embedding_output_onehot, embedding_output_genotype])

    # Additional blocks of Dense, Dropout and BatchNormalization Layers
    concatenated = Dense(64, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.2, seed=SEED)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(1024, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.2, seed=SEED)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(256, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.2, seed=SEED)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(32, kernel_initializer='glorot_normal', activation='tanh')(concatenated)

    # Output layer
    output = Dense(3, activation='linear')(concatenated)

    # Deifine model
    model = Model(inputs=[num_input, onehot_input, genotype_input], outputs=output)

    # Compile the model
    optimizer =  Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r_squared])

    return model




