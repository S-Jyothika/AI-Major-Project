## Heart Disease Prediction Using Machine Learning

-> his project predicts whether a patient has heart disease using supervised Machine Learning classification models.  
-> The goal is to compare different ML algorithms and identify the best-performing model for medical risk prediction.
-> The project is implemented in Python (Google Colab) using Scikit-learn.

## Project Overview
-> Heart disease is one of the leading causes of death worldwide. Early diagnosis can significantly improve survival rates.  
-> This project analyzes patient medical parameters such as:
      - Age  
      - Sex  
      - Chest Pain (cp)  
      - Resting Blood Pressure  
      - Cholesterol  
      - Fasting Blood Sugar  
      - Resting ECG  
      - Maximum Heart Rate  
      - Exercise-induced Angina  
      - Oldpeak  
      - Slope  
      - CA  
      - Thal  
      - Target (1 = Disease, 0 = No Disease)
-> The objective is to classify whether a patient is likely to have heart disease.

## Technologies Used
   --> Python  
   --> Google Colab  
   --> Pandas, NumPy  
   --> Matplotlib, Seaborn  
   --> Scikit-learn  

## Dataset Details
-> The dataset contains 303 patient records with 14 medical attributes.
-> Dataset used (from project PDF):  
      -- https://colab.research.google.com/drive/1USPxPSgtISPHY3zva0xM3xyBJyqBCx6j#scrollTo=IxcY9RQiGP1X
      
## Project Workflow
1. Data Import & Cleaning
   --> Loaded CSV file  
   --> Checked structure, missing values, data types  
   --> Filled missing values  
   --> Correlation analysis  

2. Exploratory Data Analysis
   --> Heatmap  
   --> Distribution visualizations  
   --> Relationship between variables  

3. Data Preprocessing
   --> Train/Test Split (80/20)  
   --> Standard Scaling  
   --> Handling categorical features  

4. Model Training
=> Machine Learning models used:
   --> Logistic Regression 
   --> Decision Tree Classifier  
   --> Random Forest Classifier  
   --> Neural Network (MLP Classifier)  

5. Model Evaluation Metrics
   --> Accuracy  
   --> Precision  
   --> Recall  
   --> F1-Score  
   --> ROC-AUC  
   --> Confusion Matrix  

## Results 
=> These values are **taken directly from the executed Colab notebook.**
   -> Logistic Regression Accuracy  : 78.68%
   -> Decision Tree Accuracy        : 70.49%
   -> Random Forest Accuracy        : 81.96% 
   -> Neural Network (MLP) Accuracy : 78.68%

## Detailed Model Metrics
   => Logistic Regression
         - Accuracy  : 0.7868
         - Precision : 0.7631  
         - Recall    : 0.8787  
         - F1 Score  : 0.8169  

   =>  Decision Tree
         - Accuracy  : 0.7049
         - Precision : 0.7027  
         - Recall    : 0.7878  
         - F1 Score  : 0.7428  

   => Random Forest
         - Accuracy  : 0.8196
         - Precision : 0.7619  
         - Recall    : 0.9696  
         - F1 Score  : 0.8533  

   => Neural Network (MLP)
         - Accuracy  : 0.7868 
         - Precision : 0.7941  
         - Recall    : 0.8181  
         - F1 Score  : 0.8059  

## Future Enhancements
   -> Hyperparameter tuning (GridSearchCV)
   -> Add XGBoost and SVM
   -> Build a Streamlit web app for real-time prediction
   -> Add SHAP feature importance analysis

## Author
Jyothika Sigirisetty
GitHub: S-Jyothika
AI / ML Developer

