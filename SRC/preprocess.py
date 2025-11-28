"""
This file contains preprocessing functions for:
- Loading dataset
- Cleaning data
- Train-test split
- Feature scaling
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Load the heart disease dataset.
    """
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """
    Clean dataset by filling missing numeric values with mean.
    """
    df = df.copy()
    df.fillna(df.mean(), inplace=True)
    return df

def split_data(df):
    """
    Split dataset into features (X) and target (y).
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def train_test_split_data(X, y):
    """
    Perform 80-20 train-test split.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test):
    """
    Scale dataset using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
