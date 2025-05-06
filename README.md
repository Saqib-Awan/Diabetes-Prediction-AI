# Code and data for AI-based diabetes prediction (PeerJ submission)

# Enhanced Diabetes Prediction Accuracy through Cutting-Edge Deep Learning and Ensemble Machine Learning Techniques

## Description

This project presents a highly accurate and generalizable approach to diabetes prediction using ensemble machine learning and deep learning models. The proposed system integrates Bagging, Boosting, Random Forest, Stacking, DNN, and LSTM architectures, achieving up to 99.90% testing accuracy and a ROC-AUC score of 1.00. It is designed to assist early diagnosis of diabetes using publicly available structured health data and advanced preprocessing strategies.

## Dataset Information

**Source**: [Pima Indian Diabetes Dataset (PIDD)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Description**: The dataset contains 768 instances of female patients aged 21 and older with 8 health-related features and a binary class label (1 = diabetic, 0 = non-diabetic).

### Preprocessing Steps
- Missing value imputation (mean for numerical, mode for categorical)
- Handling class imbalance using synthetic sampling (e.g., SMOTE)
- Data augmentation (expanded dataset from 768 to 10,000 samples)
- Feature scaling using `StandardScaler`
- Polynomial feature engineering
- 80/20 train-test split
- Final data integrity checks and shuffling

## Code Information
Below are all the functions used in the code 

- `data_preprocessing`: Loads and cleans the dataset, handles missing values, performs augmentation, scaling, and feature engineering.
- `train_ensemble_models`: Trains and evaluates Bagging, Boosting, Random Forest, and Stacking models using cross-validation.
- `train_deep_models`: Implements and trains DNN and LSTM architectures using TensorFlow/Keras.
- `evaluate_models`: Generates classification reports, confusion matrices, ROC-AUC plots, and accuracy comparisons.
- `hyperparameter_tuning`: Performs GridSearchCV and RandomizedSearchCV for model optimization.
- `model_saving`: Saves trained models using `joblib` for deployment.
- `utils`: Contains reusable functions for metrics calculation and visualizations.

## Usage Instructions

1. Clone the repository or download the ZIP:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-ai
   cd diabetes-prediction-ai

##  Files Included

- `PeerJ_Diabetes_Full_Code.ipynb` – Jupyter notebook with full code and all implementation steps.
- `diabetes.csv` – Original dataset (ensure it's in the same directory).
- `README.md` – This instruction file.

