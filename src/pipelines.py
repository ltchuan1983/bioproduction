
"""

Pipelines Module

This module provides various pipelines for loading data, preprocessing data and fitting regressor


Functions:
    create_preprocessor():
    create_pipe(regressor):
    load_sql_data(table_name):
    save_sql_data(df, table_name="table"):
    load_split_XY(table):
    load_split_save_sql_data(table_name, test_size=0.3):
    load_split_preprocess(table_name):
    load_and_augment(table_name):
    check_tables_in_db():





"""


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

        Examples of table_name
        "cleaned_data": Containing data that is already corrected and imputed
        "train_data": Train dataset. Same format as cleaned_data
        "test_data": Test dataset. Same format as cleaned_data
    
    """

import pandas as pd
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline

from helper import create_gene_list, count_genes, convert_to_num_lists, tally_num_lists, onehot_encode, remove_features

#### KEY VARIABLES

# Hardcode filepath for database. To be updated with argpase later
# If sqlite3 conn not closed properly, may need to restart the whole kernel to kill the "orphaned" connection. Else may hit OperationError: database is locked

DB_PATH = "../input/data.sqlite"

gene_string_features = ["strain_background_genotype", "genes_modified"]

gene_numlist_features = ['strain_background_genotype_modification', 'gene_deletion', 'gene_overexpression', 
    'heterologous_gene', 'replication_origin', 'codon_optimization','sensor_regulator', 
    'enzyme_redesign_evolution', 'protein_scaffold']

CATEGORICAL_FEATURES = ['reactor_type', 'media', 'oxygen']

INFO_FEATURES = ['paper_number', 'strain_background', 'product_name']
CARBON_SOURCES_FEATURES = ['cs1', 'cs2', 'cs3', 'cs3_mw', 'cs_conc3', 'CS_C3', 'CS_H3', 'CS_O3']

TARGETS = ['yield', 'titer', 'rate']

#########################################
#### Functions to create pipeline objects
#########################################

"""
    Preprocessing Pipeline:
    _______________________

    Assume new test data will also be provided in the same form as the existing data in data.sqlite3

    1. Gene_Lister: Turn strings in ["strain_background_genotype", "genes_modified"] into list of strings
    2. Gene_Counter: Count number of genes in the columns in step 1
    3. Eng_Gene_NumLister: Convert the following columns from strings into actual lists
    
    ['strain_background_genotype_modification', 'gene_deletion', 'gene_overexpression', 
    'heterologous_gene', 'replication_origin', 'codon_optimization','sensor_regulator', 
    'enzyme_redesign_evolution', 'protein_scaffold']

    4. Eng_Gene_Tally: Sum up numbers in the lists for the columns in step 2 (except gene deletion and protein scaffold whose values are all zero))
    5. Perform one-hot encoding on the following columns

    ['reactor_type', 'media', 'oxygen']

    6. Feature_Remover: Drop the following columns

    Info_features: ['paper', 'product_name']
    Carbon_sources_columns: ['cs1', 'cs2', 'cs3', 'cs3_mw', 'cs_conc3', 'CS_C3', 'CS_H3', 'CS_O3']
    Gene string columns: ["strain_background_genotype", "genes_modified"]
    Gene number columns: ['strain_background_genotype_modification', 'gene_deletion', 'gene_overexpression', 
    'heterologous_gene', 'replication_origin', 'codon_optimization','sensor_regulator', 
    'enzyme_redesign_evolution', 'protein_scaffold']


"""

def create_preprocessor():
    """ Function to create pipeline object that can injest DataFrame and preprocess
        Preprocessing steps:
        Gene_Lister: Convert strings into lists of strings
        Gene_Counter: Count the number of genes (strings) in each list
        Eng_Gene_NumLister: Convert strings into lists of numbers
        Eng_Gene_Tally: Sum up numbers in the lists
        OneHot_Encoder: One hot encode categorical features
        Remover: Remove features deemed not useful for machine learning

    Returns:
        preprocessor (Pipeline object): Pipeline object that can be used to fit_transform training data and transformt test data
    """

    Gene_Lister = FunctionTransformer(create_gene_list, kw_args={'gene_string_features': gene_string_features})
    # count_genes uses another function count_genes_per_row in helper_py, but no need to import the latter
    Gene_Counter = FunctionTransformer(count_genes, kw_args={'gene_string_features': gene_string_features})
    # convert_to_num_lists uses another function convert_to_num_list_per_row in helper_py, but no need to import the latter
    Eng_Gene_NumLister = FunctionTransformer(convert_to_num_lists, kw_args={'gene_numlist_features': gene_numlist_features})

    Eng_Gene_Tally = FunctionTransformer(tally_num_lists, kw_args={'gene_numlist_features': gene_numlist_features})

    OneHot_Encoder = FunctionTransformer(onehot_encode, kw_args={'categorical_features': CATEGORICAL_FEATURES})

    FEATURES_TO_DROP = INFO_FEATURES + CARBON_SOURCES_FEATURES + gene_string_features + gene_numlist_features + ['gene_deletion_num', 'protein_scaffold_num']
    Remover =  FunctionTransformer(remove_features, kw_args={'features_to_drop': FEATURES_TO_DROP})

    preprocessor = Pipeline([
        ('gene_lister', Gene_Lister),
        ('gene_counter', Gene_Counter),
        ('eng_gene_numlister', Eng_Gene_NumLister),
        ('eng_gene_tally', Eng_Gene_Tally),
        ('onehot', OneHot_Encoder),
        ('remover', Remover)
        # ('categorical_column', ColumnTransformer(
        #                 transformers=[('one_hot', OneHotEncoder(), CATEGORICAL_FEATURES)], remainder='passthrough')
        #             ),
        # ('to_DataFrame', FunctionTransformer(lambda x: pd.DataFrame(x)))

    ])

    return preprocessor

def create_pipe(regressor):
    """ Function to create pipeline that can ingest, preprocess and fit a specified regressor on data provided

    Args:
        regressor (regressor object): Regressor object that can be inserted into a pipe

    Returns:
        pipe [pipeline object]: Pipe containing both preprocessor and regressor
    """

    Preprocessor = create_preprocessor()

    pipe = Pipeline([
        ('preprocessor', Preprocessor),
        ('scaler', StandardScaler()),
        ('rgs', regressor)
    ])

    return pipe


#################################################
#### Functions for loading data from sqlite3 db
#################################################


def load_sql_data(table_name):
    """ Load data from specified table in database into a pandas DataFrame and print number of data points

    Args:
        table_name (str): Name of table to load data from
    
    Returns: 
        DataFrame: Data from table
    """
    
    # Connect to database
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
    """ Save DataFrame into table in sqlite3 database. Will replace table if a table of the same name already exists

    Args:
        df (DataFrame): DataFrame containing the data to save
        table_name (str): Name of the table to be saved into database. Default value "table"

    """

    database = DB_PATH
    conn = sqlite3.connect(database)
    df.to_sql(name=table_name, con=conn, if_exists="replace")
    conn.commit()
    conn.close()

def load_split_XY(table):
    """ Load data from table in sqlite3 database and split into features and targets
        Target columns predefined by global variable TARGETS

    Args:
        table (str): Name of table in database to load data from

    Returns:
        X (DataFrame), y (DataFrame): Features and targets dataframes respectively
    """

    df = load_sql_data(table)

    X = df.drop(columns=TARGETS, axis=1)
    y = df[TARGETS]

    return X, y

def load_split_save_sql_data(table_name, test_size=0.3):

    """ Load data from specified table, split into train and test sets, save each into database under
        "train_data" and "test_data"

    Args:
        table_name (str): Name of table to load data from
        test_size (float): Fraction of loaded data to be assigned to test set
    """

    # cleaned_data: No embedding
    # cleaned_data_2: Embedding (ave) for strain_background_genotype
    # cleaned_data_3: Embedding (concat) for strain_background_genotype

    df = load_sql_data(table_name)

    # train test split data
    train_df, test_df  = train_test_split(df, test_size=test_size, random_state=33)

    # Save data
    save_sql_data(train_df, "train_data")
    save_sql_data(test_df, "test_data")

def load_split_preprocess(table_name):
    """ Load from specified table and split into train and test data.
        Check all tables in db.
        Load train data and split into features and targets
        Load test data and split into features and targets
        Preprocess features (e.g. aggregate list of numbers into sum)

    Args:
        table_name (str): Name of table to load data from

    Returns:
        X_train (DataFrame), y_train (DataFrame), X_test (DataFrame), y_test (DataFrame)
    """
    
    # Prepare and save train and test 
    load_split_save_sql_data(table_name, test_size=0.3)
    tables = check_tables_in_db()

    # Load train and test data from sqlite db. Still needs further preprocessing.
    X_train, y_train = load_split_XY("train_data")
    X_test, y_test = load_split_XY("test_data")

    # Preprocess X_train and X_test
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, y_train, X_test, y_test

def load_and_augment(table_name):
    """ Load data from specified table to update train_data and test_data tables
        Augment the data by creating new rows with slightly adjusted yield, titer and rates (within t %) from each point
        Augmentation controlled by AUGMENT_NUM and MIN_ADJUST / MAX_ADJUST
        Augmented data split into features and targets.
        Preprocess features (e.g. aggregate list of numbers into sum)
        Return augmented data can be fed into subsequent modeling fitting

    Args:
        table_name (str): Name of table to load data from

    Returns:
        X_train (DataFrame), y_train (DataFrame), X_test (DataFrame), y_test (DataFrame)
    """

    # Prepare and save train and test 
    load_split_save_sql_data(table_name, test_size=0.3)
    tables = check_tables_in_db()

    # Load train and test data from sqlite db. Still needs further preprocessing.
    df_train = load_sql_data("train_data")
    df_test = load_sql_data("test_data")

    AUGMENT_NUM=20
    MIN_ADJUST = -0.01
    MAX_ADJUST=0.01 # 1%
    LEN = len(df_train)

    augmented_df_train = df_train.copy()

    for i in range(AUGMENT_NUM):
        adjusted_df_train = df_train.copy()
        adjustments_y = np.random.uniform(MIN_ADJUST, MAX_ADJUST, size=LEN)
        adjustments_t = np.random.uniform(MIN_ADJUST, MAX_ADJUST, size=LEN)
        adjustments_r = np.random.uniform(MIN_ADJUST, MAX_ADJUST, size=LEN)

        adjusted_df_train['yield'] += adjusted_df_train['yield'] * adjustments_y
        adjusted_df_train['titer'] += adjusted_df_train['titer'] * adjustments_t
        adjusted_df_train['rate'] += adjusted_df_train['rate'] * adjustments_r

        augmented_df_train = pd.concat([augmented_df_train, adjusted_df_train],axis=0)
    
    augmented_df_train = augmented_df_train.reset_index(drop=True)

    augmented_X_train = augmented_df_train.drop(columns=TARGETS, axis=1)
    augmented_y_train = augmented_df_train[TARGETS]

    X_test = df_test.drop(columns=TARGETS, axis=1)
    y_test = df_test[TARGETS]

    # Preprocess X_train and X_test
    preprocessor = create_preprocessor()
    augmented_X_train = preprocessor.fit_transform(augmented_X_train)
    X_test = preprocessor.transform(X_test)

    return augmented_X_train, augmented_y_train, X_test, y_test


def check_tables_in_db():
    """ Check, print and return the list of tables found in database

    Returns:
        tables (list): List of strings specifying the names of existing tables
    """
    
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

    return tables

