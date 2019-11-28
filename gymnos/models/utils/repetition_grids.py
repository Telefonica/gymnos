import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

############################################# RANDOM FOREST
RANDOM_FOREST_GRID = {'n_estimators': [200, 500],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'max_depth': [4, 5, 6, 7, 8],
                      'criterion': ['gini', 'entropy']}

# Number of trees in random forest
n_estimators = np.geomspace(10, 250, num=8).astype(int)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = np.geomspace(10, 250, num=8).astype(int)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
RANDOM_FOREST_RANDOM_GRID = {'n_estimators': n_estimators,
                             'max_features': max_features,
                             'max_depth': max_depth,
                             'min_samples_split': min_samples_split,
                             'min_samples_leaf': min_samples_leaf,
                             'bootstrap': bootstrap}

############################################# LIGHT GBM
LIGHT_GBM_GRID = {'n_estimators': [1000, 1500, 2000, 2500],
                  'max_depth': [4, 5, 8, -1],
                  'num_leaves': [15, 31, 63, 127],
                  'subsample': [0.6, 0.7, 0.8, 1.0],
                  'colsample_bytree': [0.6, 0.7, 0.8, 1.0]}

LIGHT_GBM_RANDOM_GRID = {'n_estimators': sp_randint(1000, 2500),
                         'max_depth': [4, 5, 8, -1],
                         'num_leaves': [15, 31, 63, 127],
                         'subsample': sp_uniform(0.6, 0.4),
                         'colsample_bytree': sp_uniform(0.6, 0.4)}

############################################# ADA BOOST
ADA_BOOST_GRID = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1]}
ADA_BOOST_RANDOM_GRID = {'n_estimators': [500, 1000, 2000], 'learning_rate': [.001, 0.01, .1]}

############################################# SVM
SVM_GRID = {'kernel': ('linear', 'rbf'),
            'C': (1, 0.25, 0.5, 0.75),
            'gamma': (1, 2, 3, 'auto'),
            'decision_function_shape': ('ovo', 'ovr'),
            'shrinking': (True, False)}
SVM_RANDOM_GRID = {'kernel': ('linear', 'rbf'),
                   'C': np.geomspace(0.1, 4, num=8),
                   'gamma': (1, 2, 3, 'auto'),
                   'decision_function_shape': ('ovo', 'ovr'),
                   'shrinking': (True, False)}

############################################# KNN
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
KNN_GRID = {'n_neighbors': k_range,
            'weights': weight_options}
KNN_RANDOM_GRID = {'n_neighbors': k_range,
                   'weights': weight_options}

############################################# XGBOOST
XGBOOST_GRID = {'Classifier__max_depth': [2, 4, 6],
                'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}

XGBOOST_RANDOM_GRID = {'Classifier__max_depth': [2, 4, 6],
                       'Classifier__n_estimators': np.geomspace(50, 500, num=5).astype(int)}
