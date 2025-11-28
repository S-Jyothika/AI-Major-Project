"""
This file trains 4 ML models:
- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network
It evaluates them, prints metrics, compares accuracy,plots confusion matrix (Random Forest), and saves model.
"""

from preprocess import (
    load_data,
    clean_data,
    split_data,
    train_test_split_data,
    scale_features
)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Train & Evaluate Models
def evaluate_model(name, y_true, y_pred):
    """
    Print evaluation metrics for a given model.
    """
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

if __name__ == "__main__":
    # 1. Load dataset
    df = load_data("dataset/heart.csv")
  
    # 2. Clean dataset
    df = clean_data(df)

    # 3. Split into features and labels
    X, y = split_data(df)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 5. Scaling
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 6. Train models
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    nn = MLPClassifier(max_iter=1000)

    lr.fit(X_train_scaled, y_train)
    dt.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    nn.fit(X_train_scaled, y_train)

    # 7. Predictions
    lr_pred = lr.predict(X_test_scaled)
    dt_pred = dt.predict(X_test_scaled)
    rf_pred = rf.predict(X_test_scaled)
    nn_pred = nn.predict(X_test_scaled)

    # 8. Evaluation
    evaluate_model("Logistic Regression", y_test, lr_pred)
    evaluate_model("Decision Tree", y_test, dt_pred)
    evaluate_model("Random Forest", y_test, rf_pred)
    evaluate_model("Neural Network", y_test, nn_pred)

    # 9. Accuracy Comparison Table
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network']
    acc = [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, nn_pred)
    ]

    print("\n===== Model Accuracy Comparison =====")
    for m, a in zip(models, acc):
        print(f"{m}: {a:.4f}")

    # 10. Confusion Matrix (Random Forest)
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
