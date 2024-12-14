#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from joblib import load
from transformers import MissingHandler, FeatureExtractor, ScalerTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier  # or whichever model you ended up choosing

def load_data(data_path):
    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-1)*512] = data
    
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test

def write_submission(y, submission_path='final_submission.csv'):
    parent_dir = os.path.dirname(submission_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(submission_path):
        os.remove(submission_path)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))
    
    # Write submission file
    with open(submission_path, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')
        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    print(f'Submission saved to {submission_path}.')



if __name__ == '__main__':
    X_train_flat, y_train, X_test_flat = load_data(data_path='./given documents/')

    n_sensors = 31
    timepoints = 512

    # Reshape the training and test data
    X_train = X_train_flat.reshape(-1, n_sensors, timepoints)
    X_test = X_test_flat.reshape(-1, n_sensors, timepoints)

    best_model = load('best_model_indicator.joblib')

    y_test = best_model.predict(X_test)

    # Write the submission file
    write_submission(y_test, submission_path='test_submission_indicator.csv')
