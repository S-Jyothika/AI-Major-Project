# Heart Disease Prediction Using Machine Learning

## Project Overview

This project predicts whether a patient has heart disease (1 = Yes, 0 = No) using medical attributes such as age, cholesterol, chest pain type, blood pressure, and other clinical parameters.The model helps in early diagnosis and clinical decision-making. The project was implemented in **Python (Google Colab)** using **four ML models**:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Neural Network (MLPClassifier)
    
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
      -> Displayed first few rows
      -> Checked data types & structure
  # 2. Data Exploration & Cleaning
      -> Checked missing values
      -> Cleaned categorical and numerical columns
      -> Removed inconsistencies
      -> Skipped unnecessary fields
  # 3. Exploratory Data Analysis (EDA)
    Visualizations included:
      -> Correlation heatmap
      -> Count of heart disease vs no disease
      -> Cholesterol & age pattern 
      -> Heart rate & chest pain analysis
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

## Confusion Matrix (Random Forest)
Even though Random Forest was not the highest accuracy model, its confusion matrix was used because:
  * It is easy to interpret
  * Random Forest gives robust classification boundaries
  * Medical ML studies commonly use Random Forest for visual evaluation
 Confusion Matrix Output:
  ```
   [[TN  FP]
   [FN  TP]]
  ```
The confusion matrix clearly shows true/false predictions for heart disease classification.

## How to Run the Project
  1. Open the notebook in Google Colab
  2. Upload the dataset `heart.csv`
  3. Install necessary Python libraries
  4. Run each cell sequentially
  5. Evaluate accuracy and confusion matrix

## Future Enhancements
  * Hyperparameter tuning for Random Forest & Neural Network
  * Adding XGBoost & SVM for better performance
  * Deploying model using Streamlit for real-time prediction
  * Integrating SHAP values for explainable AI (XAI)
  * Using cross-validation for more reliable accuracy

## Conclusion
  >> This project successfully predicts heart disease using machine learning.
  >> The best performing models were **Logistic Regression** and **Neural Network**, both achieving **85.24% accuracy**.
  >> Random Forest, while slightly less accurate, provided the **confusion matrix visualization** used in the project.
  >> This model can assist doctors in preliminary screening and risk assessment.

## Author
Jyothika Sigirisetty
AI/ML Student – Major Project
Lakireddy Balireddy College of Engineering


Just tell me!
