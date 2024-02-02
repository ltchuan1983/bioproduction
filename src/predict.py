import pandas as pd

from helper import load_sql_data, load_split_save_sql_data, preprocess_data, check_tables_in_db
from pipelines import create_pipe

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


if __name__ == "__main__":

    # Prepare and save train and test 
    load_split_save_sql_data(test_size=0.3)

    # Load cleaned data from sqlite db. Cleaned data still needs further preprocessing.
    df = load_sql_data("cleaned_data")
    
    print(df.head())

    pipe = create_pipe()

    processed_df = pipe.fit_transform(df)

    print(processed_df.info())

    check_tables_in_db()

    train_df = load_sql_data("train_data")
    test_df = load_sql_data("test_data")

    print("No. of datapoints in train data: ", len(train_df))
    print("No. of datapoints in test data: ", len(test_df))