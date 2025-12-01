"""
This file handles:
    Missing value treatment
    Splitting dataset into training & testing sets
    Feature scaling (StandardScaler)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Cleans and preprocesses the dataset.
    """
    # Fill missing numeric values with column mean
    df = df.fillna(df.mean())

    # X = all features except target
    X = df.drop("target", axis=1)

    # y = label we want to predict
    y = df["target"]

    # Splitting into train and test sets (20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardizing numeric features for ML models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
