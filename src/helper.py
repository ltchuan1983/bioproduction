import sys
import argparse
import sqlite3
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from keras import backend as K
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten, Concatenate, Embedding
from keras.optimizers import Adam
from keras import regularizers



# Values to be transferred to config file
TARGETS = ['yield', 'titer', 'rate']
NUM_FOLDS = 8

def parse_args():
    """Create ArgumentParser, set positional arguments, parse and return command line arguments

    Returns:
        args: str
            String describing the training mode to run
    """
    # Create ArgmentParser instance
    parser = argparse.ArgumentParser(
        prog = "training_and_prediction_options",
        description = "Options to use different pipelines for model training and to make predictions"
    )

    mode_choices = ['train', 'train_multi', 'train_gridsearch', 'train_bayes', 'train_nn', 'train_embed_nn', 'train_embed_genotype_nn', 'train_tunable_nn', 'train_automl', 'train_stack', 'train_stack_nn_catboost', 'train_augmentdata']

    # Add positional argument
    parser.add_argument('mode', type=str, choices=mode_choices, metavar='mode', help=f'Mode of operation. Choose one of the following {mode_choices}')

    # Parse the command line arguments

    try:
        args = parser.parse_args()
        
    except SystemExit:
        print("Options for mode:")
        print("------------------")
        print("train -----> Fit CatBoostRegressor on train data and perform predictions on test data")
        sys.exit(1)
  
    return args






def remove_data_by_mw(df, mw):
    """
    Remove data rows where mw matches the value of mw passed in.

    A function to call by passing df and features 
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 

    Parameters:
    ___________
    df: pandas DataFrame
        DataFrame containing data to be processed
    
    mw: int
        Molecular weight of product (e.g. 2 for hydrogen) 

    Returns:
    ________
    df: DataFrame where rows specified by mw have been removed
    
    """

    df = df.drop(df[df.mw==mw].index, axis=0)

    return df


def create_gene_list(df, gene_string_features):
    """
    A function to call by passing df and features 
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 
    """
    for feature in gene_string_features:
        #new_feature = gene_list_feature + "_num"
        for index, row in df.iterrows():
            gene_list = row[feature].split(', ')
            df.at[index, feature] = gene_list
    return df

def count_genes_per_row(list):
    """
    A function to be passed into df.apply()
    NOT a function to call directly
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
    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 
    """
    for feature in gene_string_features:
        tally_name = feature + '_num'
        df[tally_name] = df[feature].apply(count_genes_per_row)
    
    return df

def convert_to_num_list_per_row(string_list):
    """ Function to convert string version of num list to actual num lists and handle errors if any

    A function to be passed into df.apply()
    NOT a function to call directly

    Parameter:
    _________
    string_list: str
        string version of num list in each cell to be converted back into actual num list
    
    Returns:
    _______
    
    list of numbers corresponding to the genes involved OR None 
    """
    try:
        return eval(string_list)
    except (SyntaxError, ValueError):
        return None

def convert_to_num_lists(df, gene_numlist_features):
    """
    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 
    """
    for feature in gene_numlist_features:
        df[feature] = df[feature].apply(convert_to_num_list_per_row)
    
    return df

def convert_str_to_numpy_array(df, feature):
    for i, row in df.iterrows():
        df.at[i, feature] = np.fromstring(row[feature].strip('[]'), dtype=np.float32, sep=' ')
    
    return df

def convert_nocomma_str_to_list(df, feature):
    
    for i, row in df.iterrows():
        array_str = row[feature]
        elements = array_str.strip('[]').split(" ")
        #integer_list = [int(element) for element in elements]
        df.at[i, feature] = elements

    return df

def tally_num_lists(df, gene_numlist_features):
    """
    A function to call by passing df and features as arguments
    OR to be wrapped in FunctionTransformer and used in pipeline
    Will transform the df in place. 
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



def r2_rmse_score(y_test, y_pred, model_name):

    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])
    y_test = pd.DataFrame(y_test, columns=['yield', 'titer', 'rate'])

    print(" ")
    print("*"*50)
    print(f'Train Test Split Metrics for {model_name}')
    print("*"*50)
    print(" ")
    print('R_squared score for overall: ', r2_score(y_test, y_pred))
    print('R_squared score for yield: ', r2_score(y_test['yield'], y_pred['yield']))
    print('R_squared score for titer: ', r2_score(y_test['titer'], y_pred['titer']))
    print('R_squared score for rate: ', r2_score(y_test['rate'], y_pred['rate']))
    print(" ")
    print('RMSE for overall: ', root_mean_squared_error(y_test, y_pred))
    print('RMSE for yield: ', root_mean_squared_error(y_test['yield'], y_pred['yield']))
    print('RMSE for titer: ', root_mean_squared_error(y_test['titer'], y_pred['titer']))
    print('RMSE for rate: ', root_mean_squared_error(y_test['rate'], y_pred['rate']))

def display_cv_score(score, type):

    print("**************************************")
    print(f"Cross Validation Metrics: {type}")
    print("**************************************")
    print(score)
    print("Average: ", score.mean())

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred)))

    
def r_squared(y_true, y_pred):
    # SS_res = K.sum(K.square(y_true-y_pred))
    # SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    # return 1 - SS_res/(SS_tot + K.epsilon())

    return tf.py_function(r2_score, (y_true, y_pred), tf.float64)

def make_tunable_nn_model(hp):

    input_dim = 42
    output_dim = 3

    np.random.seed(808)
    tf.random.set_seed(808)

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

def make_nn_model_embed_cat_genotype(num_features_count, onehot_features_count, embedding_dim):

    SEED = 11
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)


    regularization_strength = 0.005


    # Define input layers
    num_input = Input(shape=(num_features_count, ), name='num_input')
    onehot_input = Input(shape=(onehot_features_count, ), name='onehot_input')
    # tokenized genotypes padded to length of 20
    genotype_input = Input(shape=(20, ), name="genotype_input")

    # Define embedding layer for strain_background_genotype_tokenized
    embedding_output_genotype = Embedding(input_dim=199, output_dim=40, input_length=20, embeddings_regularizer=regularizers.l2(regularization_strength))(genotype_input)
    embedding_output_genotype = Flatten()(embedding_output_genotype)

    # Define embedding layer for one-hot features
    # First step will produce output with shape (batch size, input_length, output_dim) (each dense vector represents each column)
    embedding_output_onehot = Embedding(input_dim=onehot_features_count, output_dim=embedding_dim, input_length=onehot_features_count)(onehot_input)
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
    optimizer =  Adam(learning_rate=0.00025)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r_squared])

    return model




