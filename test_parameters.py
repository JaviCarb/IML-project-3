from itertools import combinations
import numpy as np
import pandas as pd
import os
from joblib import dump
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GridSearchCV, LeavePOut
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


from transformers import MissingHandler, FeatureExtractor, ScalerTransformer

############################################
# Data Loading
############################################

def load_data(path, set_type='LS'):
    """
    Loads the dataset from the given directory.
    
    Parameters:
    -----------
    path : str
        Path to the directory containing the LS/TS subdirectories.
    set_type : str
        'LS' for learning set, 'TS' for test set.
        
    Returns:
    --------
    data : np.ndarray
        Shape: (n_samples, n_sensors, n_timepoints)
    subject_ids : np.ndarray
    activity_labels : np.ndarray or None
    """
    # Directory structure assumed: path/LS/..., path/TS/...
    data_path = os.path.join(path, set_type)
    
    sensor_ids = range(2, 33)
    sensor_data_list = []
    for sid in sensor_ids:
        filename = f"{set_type}_sensor_{sid}.txt"
        file_path = os.path.join(data_path, filename)
        sensor_array = np.loadtxt(file_path)
        sensor_data_list.append(sensor_array)
    
    data = np.stack(sensor_data_list, axis=1)
    data[data == -999999.99] = np.nan

    subject_ids = np.loadtxt(os.path.join(data_path, 'subject_Id.txt')).astype(int)

    activity_labels = None
    if set_type == 'LS':
        activity_labels = np.loadtxt(os.path.join(data_path, 'activity_Id.txt')).astype(int)
    
    return data, subject_ids, activity_labels





def leave_two_subjects_out(subject_ids):
    """
    Generator that yields (train_index, test_index) for leave-two-subjects-out splits.
    
    Parameters
    ----------
    subject_ids : array-like
        A 1D array containing the subject ID for each sample.
    
    Yields
    ------
    train_index, test_index : np.ndarray
        Arrays of indices for training and test sets.
    """
    # Get unique subjects
    unique_subjects = np.unique(subject_ids)
    
    # Generate all combinations of two subjects for test
    for test_subjects in combinations(unique_subjects, 2):
        # Boolean mask for test samples
        test_mask = np.isin(subject_ids, test_subjects)
        test_index = np.where(test_mask)[0]
        
        # The rest of the subjects are training
        train_index = np.where(~test_mask)[0]
        
        yield train_index, test_index





############################################
# Example Pipeline and Parameter Grid
############################################

# Load training data
path = "./given documents"  # Adjust to your data directory
data, subject_ids, activity_labels = load_data(path, set_type='LS')

# Identify samples with any missing sensor
missing_mask = np.any(np.isnan(data), axis=2)  # shape (n_samples, n_sensors)
samples_to_drop = np.any(missing_mask, axis=1)

# Drop these samples from data, subject_ids, and activity_labels
X_clean = data[~samples_to_drop]
y_clean = activity_labels[~samples_to_drop]
subjects_clean = subject_ids[~samples_to_drop]


pipeline = Pipeline(steps=[
    ('missing_handler', MissingHandler(method='impute', impute='median')),
    ('feature_extractor', FeatureExtractor(segment=True, n_segments=4, feature_level='extended')),
    ('scaler', ScalerTransformer(method='standard')),
    ('pca', 'passthrough'),
    ('clf', RandomForestClassifier(random_state=0))
], verbose=False)

#Perceptron(max_iter=1000, tol=1e-3)
#DecisionTreeClassifier(max_depth=None)
#KNeighborsClassifier(n_neighbors=1)

#HistGradientBoostingClassifier()
#RandomForestClassifier()

# Without dropping missing values
param_grid = {
    'missing_handler__method': ['indicator'],
    'missing_handler__impute': ['mean'],
    'feature_extractor__segment': [False],
    'feature_extractor__n_segments': [4],
    'feature_extractor__feature_level': ['extended'],
    'pca': ['passthrough'], #PCA(n_components=10)
    'scaler__method': ['robust'],
    'clf__n_estimators': [300],   # RandomForest hyperparams
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [4]
}

#For KNN:
#'clf__n_neighbors': [1, 2, 3],

#For DecisionTree:
#'clf__max_depth': [10, None],

#For Perceptron:
#'clf__max_iter': [500, 1000, 2000],
#'clf__eta0': [1.0, 0.1, 0.01],  # Initial learning rate
#'clf__penalty': [None, 'l2', 'l1', 'elasticnet']

#For HistGradientBoosting:
#'clf__max_iter': [100, 200],             # HistGradientBoostingClassifier hyperparams
#'clf__learning_rate': [0.1, 0.01],
#'clf__max_depth': [None, 10],            # None means no maximum depth
#'clf__l2_regularization': [0.0, 1.0]

#For RandomForest:
#'clf__n_estimators': [100, 200, 300, 400],   # RandomForest hyperparams
#'clf__max_depth': [None, 5, 10, 15, 20],
#'clf__min_samples_split': [2, 3, 5],
#'clf__min_samples_leaf': [1, 2, 4, 6, 8]


gkf = GroupKFold(n_splits=5)
my_cv = list(leave_two_subjects_out(subject_ids))

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=gkf.split(data, activity_labels, groups=subject_ids),
    scoring='accuracy',
    n_jobs=10,
    verbose=10
)
grid_search.fit(data, activity_labels)

print("\n\nFirst Search Done!")
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
print("\n\n")

#Save the model using joblib
dump(grid_search.best_estimator_, 'best_model_indicator.joblib')

# With dropping missing values
param_grid_2 = {
    'missing_handler': ['passthrough'],
    'feature_extractor__segment': [False],
    'feature_extractor__n_segments': [4],
    'feature_extractor__feature_level': ['extended'],
    'pca': ['passthrough'], #PCA(n_components=10)
    'scaler__method': ['robust'],
    'clf__n_estimators': [300],   # RandomForest hyperparams
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [4]
}

grid_search2 = GridSearchCV(
    pipeline,
    param_grid_2,
    cv=gkf.split(X_clean, y_clean, groups=subjects_clean),
    scoring='accuracy',
    n_jobs=10,
    verbose=10
)
grid_search2.fit(X_clean, y_clean)

print("\n\nSecond Search Done!")
print("Best params:", grid_search2.best_params_)
print("Best CV score:", grid_search2.best_score_)

#Save the model using joblib
dump(grid_search2.best_estimator_, 'best_model_drop.joblib')