
# ============================
# COLAB SETUP (Run only if in Colab)
# ============================
import sys
import os
from pathlib import Path

try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive')

    # Adjust this path if needed
    project_path = Path('/content/drive/MyDrive/predicting_cust_churn')
    sys.path.append(str(project_path))
    os.chdir(project_path)

except ImportError:
    pass  # Not running in Colab




# For type hints
from typing import Optional

# Colab utilities
from google.colab import files

# Core third-party libraries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# LightGBM
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Scikit-learn: model selection and evaluation
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Local Utilities
from src.model_utils import tuning_cv, accuracy_calc, shap_eval



if __name__ == '__main__':

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

    # We need to make sure we have a version of LGBMClassifier that can run on GPUs
    print("LightGBM version:", lgb.__version__)

    try:
        model_light = lgb.LGBMClassifier(device='gpu', eval_metric='auc', random_state=12345)
        print("GPU support is enabled for LightGBM.")
    except Exception as e:
        print("GPU support is NOT enabled:", e)

    param_grid = {
        'num_leaves': [15, 31],                 
        'max_depth': [5, -1],                   
        'learning_rate': [0.01, 0.1],           
        'n_estimators': [100, 300],             
        'subsample': [0.8, 1.0],                
        'colsample_bytree': [0.8, 1.0],         
        'class_weight': [None, 'balanced']      
    }

    model_light = tuning_cv(features_train_light_xgb, 
                            target_train_light_xgb, 
                            model_light, 
                            param_grid)

    accuracy_calc(features_train_light_xgb, 
                  target_train_light_xgb,  
                  model_light)

    shap_eval(model_light, 
              features_train_light_xgb, 
              target_train_light_xgb)

