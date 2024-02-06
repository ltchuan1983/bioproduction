import sys
import argparse
import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from keras import backend as K
import tensorflow as tf

TARGETS = ['yield', 'titer', 'rate']
NUM_FOLDS = 8

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

# Hardcode filepath for database. To be updated with argpase later
DB_PATH = "../input/data.sqlite"

"""
    If sqlite3 conn not closed properly, may need to restart the whole kernel to kill the "orphaned" connection. Else may hit OperationError: database is locked
"""

def load_sql_data(table_name):
    """
    
    Columns of cleaned_table in data.sqlite

    ['paper_number', 'cs1', 'cs1_mw', 'cs_conc1', 'CS_C1', 'CS_H1', 'CS_O1',
       'cs2', 'cs2_mw', 'cs_conc2', 'CS_C2', 'CS_H2', 'CS_O2', 'cs3', 'cs3_mw',
       'cs_conc3', 'CS_C3', 'CS_H3', 'CS_O3', 'reactor_type', 'rxt_volume',
       'media', 'temp', 'oxygen', 'strain_background',
       'strain_background_genotype', 'strain_background_genotype_modification',
       'genes_modified', 'gene_deletion', 'gene_overexpression',
       'heterologous_gene', 'replication_origin', 'codon_optimization',
       'sensor_regulator', 'enzyme_redesign_evolution', 'protein_scaffold',
       'dir_evo', 'Mod_path_opt', 'product_name', 'no_C', 'no_H', 'no_O',
       'no_N', 'mw', 'yield', 'titer', 'rate', 'fermentation_time']
    
    """

    database = DB_PATH
    conn = sqlite3.connect(database)

    # Load data into pandas DataFrame
    query = f"""SELECT * from {table_name}"""
    df = pd.read_sql_query(query, conn, index_col="index")

    conn.commit()
    conn.close()

    print(f"No. of data points in {table_name}: ", len(df))

    return df


def save_sql_data(df, table_name="table"):
    """
        Examples of table_name
        "cleaned_data": Containing data that is already corrected and imputed
        "train_data": Train dataset. Same format as cleaned_data
        "test_data": Test dataset. Same format as cleaned_data
    """

    database = DB_PATH
    conn = sqlite3.connect(database)
    df.to_sql(name=table_name, con=conn, if_exists="replace")
    conn.commit()
    conn.close()


def load_split_save_sql_data(test_size=0.3):

    df = load_sql_data("cleaned_data")

    # train test split data
    train_df, test_df  = train_test_split(df, test_size=test_size, random_state=33)

    # Save data
    save_sql_data(train_df, "train_data")
    save_sql_data(test_df, "test_data")

def check_tables_in_db():
    
    database = DB_PATH
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    query = """SELECT name FROM sqlite_master WHERE type='table';"""

    cursor.execute(query)

    tables = cursor.fetchall()

    for index, table in enumerate(tables):
        print(f"Table {index}: ", table[0])
    
    cursor.close()
    conn.close()

def load_split_XY(table):

    df = load_sql_data(table)

    X = df.drop(columns=TARGETS, axis=1)
    y = df[TARGETS]

    return X, y


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


def r2_rmse_score2(y_test, y_pred, model_name):

    r2_result = r_squared(y_test['titer'], y_pred['titer'])

    print(" ")
    print("*"*50)
    print(f'Train Test Split Metrics for {model_name}')
    print("*"*50)
    print(" ")
    print('R_squared score for overall: ', r2_result)
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

