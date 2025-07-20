

# For type hints
from typing import Optional

# Colab utilities
from google.colab import files

# Core third-party libraries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# CatBoost
from catboost import CatBoostClassifier, Pool

# Scikit-learn: model selection and evaluation and type hints
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Local Utilities
from src.model_utils import tuning_cv, accuacy_calc, shap_eval



if __name__ == '__main__':

    train_df_cat = pd.read_csv('train_data_cat.csv')
    test_df_cat = pd.read_csv('test_data_cat.csv')

    features_train_cat = train_df_cat.drop('target', axis=1)
    target_train_cat = train_df_cat['target']

    features_test_cat = test_df_cat.drop('target', axis=1)
    target_test_cat = test_df_cat['target']

    categorical_cols = ['type',
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

    model_cat = CatBoostClassifier(random_seed=12345, eval_metric='AUC', task_type='GPU')



    param_grid = {
    'iterations': [100, 300],               # Keep it light but effective
    'depth': [4, 6],                        # Balanced tree depth options
    'learning_rate': [0.05, 0.1],           # Two sensible learning rates
    'l2_leaf_reg': [3, 7],                  # Light regularization options
    'border_count': [32, 50],               # Numerical feature binning splits
    'random_strength': [1, 2],              # Control randomness in splitting
    'bagging_temperature': [0, 1],          # Simple bagging temps to test
    'scale_pos_weight': [1, 3]}             # Basic class imbalance adjustment

    model_cat = tuning_cv(features_train_cat, 
                                   target_train_cat, 
                                   model_cat, 
                                   param_grid,
                                   cat_features=categorical_cols)

    accuracy_calc(features_train_cat, 
                  target_train_cat,  
                  model_cat)



    shap_eval(model_cat, 
              features_train_cat, 
              target_train_cat)

