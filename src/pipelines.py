"""
    Pipelines Module

    This module provides various pipelines for loading data, preprocessing data and fitting regressor

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

    Tables in db:
    "cleaned_data": Containing data that is already corrected and imputed
    "cleaned_data_2": Data where strain_background_genotype already converted to embeddings via unsupervised word2vec and averaging embedding per token
    "cleaned_data_3": Data where strain_background_genotype already converted to embeddings via unsupervised word2vec and appending embedding per token
    "cleaned_data_4": Data where strain_background_genotype only tokenized for supervised embeddings within the neural network itself
    "train_data": Train dataset. Same format as cleaned_data
    "test_data": Test dataset. Same format as cleaned_data

    Functions:
        create_preprocessor(): Create pipeline to preprocess data loaded from sqlite3 db
        create_pipe(regressor): Create pipeline covering both preprocessing and regressor fitting/predictions
        load_sql_data(table_name): Load data from specified table in sqlite3 db
        save_sql_data(df, table_name="table"): Save data to specified table in sqlite3 db. Replace if exists
        load_split_XY(table): Load data from specified table in sqlite3 db and split into features and targets
        load_split_save_sql_data(table_name, test_size=0.3):Load data from specified table in sqlite3 db, train_test_split and save to specified tables
        load_split_preprocess(table_name): Load data from specified table in sqlite3 db and preprocess with pipeline above
        load_and_augment_targets(table_name): Load data from specified table in sqlite3, preprocess and perform data augmentation (by varying target values)
        load_and_augment_features(table_name): Load data from specified table in sqlite3, preprocess and perform data augmentation (by varying feature values)
        check_tables_in_db(): Loop through every table and print its name
"""

import pandas as pd
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline

from helper import create_gene_list, count_genes, convert_to_num_lists, tally_num_lists, onehot_encode, remove_features

from config import Config

#### KEY VARIABLES

# Hardcode filepath for database. To be updated with argpase later
# If sqlite3 conn not closed properly, may need to restart the whole kernel to kill the "orphaned" connection. Else may hit OperationError: database is locked

config = Config()

DB_PATH = config.config_data['DB_PATH']
GENE_STRING_FEATURES = config.config_data['GENE_STRING_FEATURES']
GENE_NUMLIST_FEATURES = config.config_data['GENE_NUMLIST_FEATURES']
CATEGORICAL_FEATURES = config.config_data['CATEGORICAL_FEATURES']
INFO_FEATURES = config.config_data['INFO_FEATURES']
CARBON_SOURCES_FEATURES = config.config_data['CARBON_SOURCES_FEATURES']
TARGETS = config.config_data['TARGETS']

#########################################
#### Functions to create pipeline objects
#########################################

def create_preprocessor():
    """ Function to create pipeline object that can injest DataFrame and preprocess

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

        Preprocessing steps:
        ____________________
        Gene_Lister: Convert strings into lists of strings
        Gene_Counter: Count the number of genes (strings) in each list
        Eng_Gene_NumLister: Convert strings into lists of numbers
        Eng_Gene_Tally: Sum up numbers in the lists
        OneHot_Encoder: One hot encode categorical features
        Remover: Remove features deemed not useful for machine learning

    Returns:
        preprocessor (Pipeline object): Pipeline object that can be used to fit_transform training data and transformt test data
    """

    Gene_Lister = FunctionTransformer(create_gene_list, kw_args={'gene_string_features': GENE_STRING_FEATURES})
    # count_genes uses another function count_genes_per_row in helper_py, but no need to import the latter
    Gene_Counter = FunctionTransformer(count_genes, kw_args={'gene_string_features': GENE_STRING_FEATURES})
    # convert_to_num_lists uses another function convert_to_num_list_per_row in helper_py, but no need to import the latter
    Eng_Gene_NumLister = FunctionTransformer(convert_to_num_lists, kw_args={'gene_numlist_features': GENE_NUMLIST_FEATURES})

    Eng_Gene_Tally = FunctionTransformer(tally_num_lists, kw_args={'gene_numlist_features': GENE_NUMLIST_FEATURES})

    OneHot_Encoder = FunctionTransformer(onehot_encode, kw_args={'categorical_features': CATEGORICAL_FEATURES})

    FEATURES_TO_DROP = INFO_FEATURES + CARBON_SOURCES_FEATURES + GENE_STRING_FEATURES + GENE_NUMLIST_FEATURES + ['gene_deletion_num', 'protein_scaffold_num']
    Remover =  FunctionTransformer(remove_features, kw_args={'features_to_drop': FEATURES_TO_DROP})

    preprocessor = Pipeline([
        ('gene_lister', Gene_Lister),
        ('gene_counter', Gene_Counter),
        ('eng_gene_numlister', Eng_Gene_NumLister),
        ('eng_gene_tally', Eng_Gene_Tally),
        ('onehot', OneHot_Encoder),
        ('remover', Remover)
    ])

    return preprocessor

def create_pipe(regressor):
    """ Function to create pipeline that can ingest, preprocess and fit a specified regressor on data provided

    Args:
        regressor (regressor object): Regressor object that can be inserted into a pipe

    Returns:
        pipe (pipeline object): Pipe containing both preprocessor and regressor
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

    # Remove more outliers
    df.drop(df[df['yield']>1].index, axis=0, inplace=True)
    df.drop(df[df['titer']>60].index, axis=0, inplace=True)
    df.drop(df[df['rate']>1.8].index, axis=0, inplace=True)

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

def load_and_augment_targets(table_name):
    """ Load data from specified table to update train_data and test_data tables
        Augment TARGET VALUES by creating new rows with slightly adjusted yield, titer and rates (within t %) from each point
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

def load_and_augment_features(table_name):
    """ Load data from specified table to update train_data and test_data tables
        Augment select FEATURE VALUES by creating new rows with slightly adjustments (within t %) from each point
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


    # Preprocess X_train and X_test
    preprocessor = create_preprocessor()
    df_train = preprocessor.fit_transform(df_train)
    df_test = preprocessor.transform(df_test)

    features_to_augment = ['cs_conc1', 'cs_conc2', 'rxt_volume', 'temp', 'fermentation_time',
                           'strain_background_genotype_num', 'genes_modified_num', 'strain_background_genotype_modification_num',
                           'gene_overexpression_num', 'heterologous_gene_num', 'replication_origin_num', 'codon_optimization_num',
                           'sensor_regulator_num', 'enzyme_redesign_evolution_num']

    # Augment features instead of target values

    AUGMENT_NUM=config.config_data['AUGMENT_NUM']
    MIN_ADJUST = -config.config_data['AUGMENT_RANGE']
    MAX_ADJUST=config.config_data['AUGMENT_RANGE'] # e.g. 5%
    LEN = len(df_train)

    augmented_df_train = df_train.copy()

    for i in range(AUGMENT_NUM):
        adjusted_df_train = df_train.copy()
        adjustments = []
        for i, feature in enumerate(features_to_augment):

            adjustment = np.random.uniform(MIN_ADJUST, MAX_ADJUST, size=LEN)
            adjusted_df_train[feature] += adjusted_df_train[feature] * adjustment

        augmented_df_train = pd.concat([augmented_df_train, adjusted_df_train],axis=0)
    
    augmented_df_train = augmented_df_train.reset_index(drop=True)

    augmented_X_train = augmented_df_train.drop(columns=TARGETS, axis=1)
    augmented_y_train = augmented_df_train[TARGETS]

    X_test = df_test.drop(columns=TARGETS, axis=1)
    y_test = df_test[TARGETS]

    return augmented_X_train, augmented_y_train, X_test, y_test

def load_test_data(table):
    """Load data from specified table for predictions

    Args:
        table (str): Name of table containing test data

    Returns:
        X_test, y_test (pd DataFrame): Feature and target values of test data
    """
    X_test, y_test = load_split_XY(table)
    
    preprocessor = create_preprocessor()
    X_test = preprocessor.fit_transform(X_test)
    
    return X_test, y_test

def check_tables_in_db():
    """ Check, print and return the list of tables found in database

    Returns:
        tables (list): List of strings specifying the names of existing tables
    """
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """SELECT name FROM sqlite_master WHERE type='table';"""

    cursor.execute(query)

    tables = cursor.fetchall()

    for index, table in enumerate(tables):
        print(f"Table {index}: ", table[0])
    
    cursor.close()
    conn.close()

    return tables

