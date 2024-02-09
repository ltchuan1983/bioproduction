

import pandas as pd


from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from catboost import CatBoostRegressor

from helper import create_gene_list, count_genes, convert_to_num_lists, tally_num_lists, onehot_encode, remove_features
from helper import load_split_save_sql_data, check_tables_in_db, load_split_XY
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

gene_string_features = ["strain_background_genotype", "genes_modified"]

gene_numlist_features = ['strain_background_genotype_modification', 'gene_deletion', 'gene_overexpression', 
    'heterologous_gene', 'replication_origin', 'codon_optimization','sensor_regulator', 
    'enzyme_redesign_evolution', 'protein_scaffold']

CATEGORICAL_FEATURES = ['reactor_type', 'media', 'oxygen']

INFO_FEATURES = ['paper_number', 'strain_background', 'product_name']
CARBON_SOURCES_FEATURES = ['cs1', 'cs2', 'cs3', 'cs3_mw', 'cs_conc3', 'CS_C3', 'CS_H3', 'CS_O3']


def create_preprocessor():

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

    Preprocessor = create_preprocessor()

    pipe = Pipeline([
        ('preprocessor', Preprocessor),
        ('scaler', StandardScaler()),
        ('rgs', regressor)
    ])

    return pipe

def create_linear_regression_pipe(regressor):

    Preprocessor = create_preprocessor()

    pipe = Pipeline([
        ('preprocessor', Preprocessor),
        ('rgs', regressor)
    ])

    return pipe

def load_split_preprocess(table_name):

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

