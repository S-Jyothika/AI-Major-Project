"""
This file:
    Loads dataset
    Preprocesses data
    Trains 4 machine learning models
    Sends results to evaluation module
"""

from load_data import load_dataset
from preprocess import preprocess_data
from evaluate import evaluate_model

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_models():

    # STEP 1: Load dataset
    df = load_dataset("../data/heart.csv")

    # STEP 2: Preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # STEP 3: Define ML models
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    }

    # STEP 4: Train each model one by one
    for name, model in models.items():
        print(f"\n⚙ Training {name}...")
        model.fit(X_train, y_train)

        # STEP 5: Evaluate model performance
        evaluate_model(name, model, X_test, y_test)
        
