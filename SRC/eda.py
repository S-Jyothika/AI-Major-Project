"""
This file performs Exploratory Data Analysis (EDA):
  Correlation heatmap
  Understanding relationships between attributes
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    """
    Generates correlation heatmap and saves it to results folder.
    """

    # Creating a heatmap to show how features correlate
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap between Features")

    # Save heatmap image for GitHub
    plt.savefig("../results/correlation_heatmap.png")

    plt.show()
    print("\n EDA Completed (Heatmap Saved)")
