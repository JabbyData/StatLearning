""" Module to preprocess the data """
import pandas as pd


def normalize(data,a,b,feature_threshold=20):
    new_data = data.copy()
    """ Normalize the data within the interval [a,b] """
    for feature in data.columns:
        if data[feature].nunique() < feature_threshold:
            # Performs One-hot encoding
            new_data = pd.get_dummies(new_data, columns=[feature])
        else:
            # Performs normalization
            new_data[feature] = a + (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min()) * (b - a)
    return new_data


def standardize(data,feature_threshold=100):
    new_data = data.copy()
    """ Standardize the data """
    for feature in data.columns:
        if data[feature].nunique() < feature_threshold:
            # Performs One-hot encoding
            new_data = pd.get_dummies(new_data, columns=[feature])
        else:
            # Performs standardization
            new_data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
    return new_data