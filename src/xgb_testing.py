

# Standard library imports
import os

# For type hints
from typing import Optional

# Colab utilities
from google.colab import files

# Core third-party libraries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# XGBoost
import xgboost
from xgboost import XGBClassifier

# Scikit-learn: model selection and evaluation and type hints
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Local Utilities
from src.model_utils import threshold_calc, print_metrics, shap_eval



if __name__ == '__main__':

    # Make sure we have xgboost version that can handle categorical variables natively
    print(xgboost.__version__)

    train_df_light_xgb = pd.read_csv('train_data_light_xgb.csv')
    test_df_light_xgb = pd.read_csv('test_data_light_xgb.csv')

    categorical_cols = ['type',
                    'total_charges',
                    'paperless_billing',
                    'payment_method',
                    'gender', 
                    'partner',
                    'dependents',
                    'p_i_or_b',
                    'internet_service',
                    'online_security',
                    'online_backup',
                    'device_protection',
                    'tech_support',
                    'streaming_tv',
                    'streaming_movies',
                    'multiple_lines']



    features_train_light_xgb = train_df_light_xgb.drop('target', axis=1)
    target_train_light_xgb = train_df_light_xgb['target']

    features_test_light_xgb = test_df_light_xgb.drop('target', axis=1)
    target_test_light_xgb = test_df_light_xgb['target']

    features_train_light_xgb[categorical_cols] = features_train_light_xgb[categorical_cols].astype('category')
    features_test_light_xgb[categorical_cols] = features_test_light_xgb[categorical_cols].astype('category')

    # Load model if the file exists
    model_path = "model_xgb.pkl"
    if os.path.exists(model_path):
        model_xgb = joblib.load(model_path)
        print("Loaded model_xgb from file.")
    else:
        raise FileNotFoundError(f"{model_path} not found. Make sure the model was trained and saved.")


    optimal_threshold = threshold_calc(model_xgb, 
                                    features_train_light_xgb, 
                                    target_train_light_xgb)

    print_metrics(model_xgb, 
                  features_train_light_xgb, 
                  target_train_light_xgb, 
                  features_test_light_xgb, 
                  target_test_light_xgb,
                  optimal_threshold)

    shap_eval(model_xgb, 
              features_test_light_xgb, 
              target_test_light_xgb)


