"""
This file:
  Calculates all evaluation metrics
  Generates confusion matrix images
  Saves plots to results folder
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)

def evaluate_model(name, model, X_test, y_test):

    # Predict using the trained model
    y_pred = model.predict(X_test)

    # Print detailed evaluation metrics
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.tight_layout()

    # Save confusion matrix image
    file_path = f"../results/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(file_path)
    plt.close()

    print(f" Confusion matrix saved as {file_path}")
