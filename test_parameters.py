import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
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
    ('missing_handler', MissingHandler(method='impute')),
    ('feature_extractor', FeatureExtractor(segment=True, n_segments=4, feature_level='extended')),
    ('scaler', ScalerTransformer(method='standard')),
    ('pca', 'passthrough'),
    ('clf', DecisionTreeClassifier(max_depth=5))
], verbose=True, memory='./cache')

param_grid = {
    'missing_handler__method': ['indicator', 'impute'],
    'feature_extractor__segment': [True, False],
    'feature_extractor__n_segments': [4, 8],
    'feature_extractor__feature_level': ['basic', 'extended'],
    'pca': [PCA(n_components=10), PCA(n_components=100), 'passthrough'],
    'scaler__method': ['standard', 'robust'],
    'clf__max_depth': [None]
}

gkf = GroupKFold(n_splits=5)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=gkf.split(data, activity_labels, groups=subject_ids),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(data, activity_labels)

#pca = grid_search.best_estimator_.named_steps['pca']
#print("Explained variance ratio of the PCA:", pca.explained_variance_ratio_)

print("\n\nFirst Search Done!")
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)


# With dropping missing values
'''
param_grid_2 = {
    'missing_handler': ['passthrough'],
    'feature_extractor__segment': [False],
    'feature_extractor__n_segments': [4],
    'feature_extractor__feature_level': ['extended'],
    'pca': ['passthrough'],
    'scaler__method': ['robust'],
    'clf__max_depth': [None]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid_2,
    cv=gkf.split(X_clean, y_clean, groups=subjects_clean),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_clean, y_clean)

#pca = grid_search.best_estimator_.named_steps['pca']
#print("Explained variance ratio of the PCA:", pca.explained_variance_ratio_)

print("\n\nSecond Search Done!")
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
'''