

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Local Utilities
from src.model_utils import tuning_cv, accuracy_calc, shap_eval



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

    model_xgb = XGBClassifier(
    tree_method="gpu_hist",
    enable_categorical=True,        # enables native handling of categorical features
    predictor="gpu_predictor",      # optional for faster inference
    use_label_encoder=False,
    eval_metric='auc',
    random_state=12345)

    param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'scale_pos_weight': [3, 4]}  # imbalance adjustment


    model_xgb = tuning_cv(features_train_light_xgb, 
                                   target_train_light_xgb, 
                                   model_xgb, 
                                   param_grid)

    accuracy_calc(features_train_light_xgb, 
                  target_train_light_xgb,  
                  model_xgb)



    shap_eval(model_xgb, 
              features_train_light_xgb, 
              target_train_light_xgb)

    # Save model to project directory
    joblib.dump(model_xgb, 'model_xgb.pkl')

