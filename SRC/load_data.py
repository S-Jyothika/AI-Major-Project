"""
This file loads the Heart Disease dataset and prints basic information.
Useful for understanding dataset structure before preprocessing.
"""

import pandas as pd

def load_dataset(path):
    """
    Loads dataset from the given CSV path and prints information.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)

    # Display first few rows for overview
    print("\n First 5 Rows of Dataset:\n", df.head())

    # Display column info (data types, null values, etc.)
    print("\n Dataset Info:\n")
    print(df.info())

    # Check missing values in each column
    print("\n Missing Values:\n", df.isnull().sum())

    return df
