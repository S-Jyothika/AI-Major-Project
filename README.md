# Heart Disease Prediction Using Machine Learning

## Project Overview

This project predicts whether a patient has heart disease (1 = Yes, 0 = No) using medical attributes such as age, cholesterol, chest pain type, blood pressure, and other clinical parameters.The model helps in early diagnosis and clinical decision-making. The project was implemented in **Python (Google Colab)** using **four ML models**:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Neural Network (MLPClassifier)
The goal is to support early diagnosis and medical decision-making by identifying patterns associated with heart disease.

## Technologies Used
  * Python
  * Google Colab
  * Pandas, NumPy
  * Seaborn, Matplotlib
  * Scikit-learn

## Dataset Details
The dataset contains 303 patient records with the following important features:

| Feature  -- Description                               |

| Age      -- Age of the patient                        |
| Sex      -- 1 = Male, 0 = Female                      |
| Cp       -- Chest Pain Type                           |
| Trestbps -- Resting Blood Pressure                    |
| Chol     -- Cholesterol level                         |
| Fbs      -- Fasting Blood Sugar                       |
| Thalach  -- Maximum Heart Rate Achieved               |
| Exang    -- Exercise-Induced Angina                   |
| Oldpeak  -- ST Depression                             |
| Thal     -- Thalassemia                               |
| Target   -- (1 = Heart Disease, 0 = No Heart Disease) |

## Project Workflow
  # 1. Data Import & Loading
      -> Loaded dataset into Pandas
      -> Checked structure, size, column names
  # 2. Data Exploration & Cleaning
      -> Checked for missing values
      -> Cleaned numeric/categorical fields
      -> Identified important medical features
  # 3. Exploratory Data Analysis (EDA)
    Visualizations included:
      -> Correlation heatmap
      -> Count of heart disease vs no disease
      -> Cholesterol & age pattern 
      -> Chest pain vs heart disease relationship
    These visualizations helped identify which factors impact heart disease.
  # 4. Data Preprocessing
      -> Selected important medical features
      -> Performed **train-test split**
      -> Applied **StandardScaler** for normalization
  # 5. Model Training
    Four ML models were trained:
      -> Logistic Regression
      -> Decision Tree Classifier
      -> Random Forest Classifier
      -> Neural Network (MLPClassifier)
    Each model predicts whether a patient likely has heart disease.

## Evaluation Metrics Used
Models are evaluated using the following metrics:
>> Accuracy
    -> Measures the overall correctness of the model.
>> Precision
    -> Out of the predicted positives, how many were correct.
>> Recall
    -> Out of the actual positives, how many were correctly predicted.
>> F1 Score
    -> Harmonic mean of precision and recall.
>> Confusion Matrix
    -> Shows True Positive, True Negative, False Positive, and False Negative counts.
>> ROC-AUC Score
    -> Measures the ability of the model to distinguish between classes; higher AUC indicates better performance.
    -> These metrics ensure comprehensive evaluation of the model beyond just accuracy.

## Model Evaluation Results
Project results:
| Model                    --> Accuracy   |
| ------------------------ --> ---------- |
| **Logistic Regression**  --> **85.24%** |
| **Neural Network (MLP)** --> **85.24%** |
| Random Forest            --> 83.60%     |
| Decision Tree            --> 81.96%     |

## Best Performing Models:
  -- Logistic Regression
  -- Neural Network (MLP)
Both achieved the highest accuracy.

## Confusion Matrices 
The project generates separate confusion matrix heatmaps for all four models:
      >> Logistic Regression
      >> Decision Tree
      >> Random Forest
      >> Neural Network (MLP)
Confusion matrices clearly show:
      --> True Positives (TP)
      --> True Negatives (TN)
      --> False Positives (FP)
      --> False Negatives (FN)
These help compare how each model classifies heart disease cases.

## How to Run the Project
  1. Open the notebook in Google Colab
  2. Upload the dataset `heart.csv`
  3. Install necessary Python libraries
  4. Run each cell sequentially
  5. Evaluate accuracy and confusion matrix

## Future Enhancements
  * Hyperparameter tuning for better accuracy
  * Adding XGBoost & SVM for better performance
  * Deploying model using Streamlit for real-time prediction
  * Integrating SHAP values for explainable AI (XAI)
  * Using cross-validation for more reliable accuracy

## Conclusion
  >> This project successfully predicts heart disease using machine learning algorithms.
  >> The best performing models were **Logistic Regression** and **Neural Network**, both achieving **85.24% accuracy**.
  >> Confusion matrices for all four models provide clear performance comparison.
  >> This model can assist doctors in preliminary screening and risk assessment.

## Author
Jyothika Sigirisetty
AI/ML Student – Major Project
Lakireddy Balireddy College of Engineering
