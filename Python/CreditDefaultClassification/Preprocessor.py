""" Module to preprocess the data """
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer


def standardize(data,feature_threshold=80):
    """ Returns a standardized copy of the data """
    # Looking for the numerical and categorical columns
    num_cols = []
    cat_cols = []
    for col in data.columns:
        if data[col].unique().shape[0] < feature_threshold:  # arbitrary threshold so that LIMIT_BAL is not considered as a categorical variable
            cat_cols.append(col)
        else:
            num_cols.append(col)
    cat_df = data[cat_cols]
    num_df = data[num_cols]

    # Scaling of num_df
    scaler = StandardScaler()
    num_df_scaled = pd.DataFrame(scaler.fit_transform(num_df), columns=num_cols)

    # One hot encoding of cat_df
    encoder = OneHotEncoder()
    cat_df_encoded = pd.DataFrame(encoder.fit_transform(cat_df).toarray(),columns=encoder.get_feature_names_out(cat_cols))

    # Concatenation of the two dataframes
    new_data = pd.concat([num_df_scaled, cat_df_encoded], axis=1)
    return new_data


def normalize(data,feature_threshold=80):
    """ Returns a normalized copy of the data """
    # Looking for the numerical and categorical columns
    num_cols = []
    cat_cols = []
    for col in data.columns:
        if data[col].unique().shape[0] < feature_threshold:  # arbitrary threshold so that LIMIT_BAL is not considered as a categorical variable
            cat_cols.append(col)
        else:
            num_cols.append(col)
    cat_df = data[cat_cols]
    num_df = data[num_cols]

    # Scaling of num_df
    scaler = Normalizer()
    num_df_scaled = pd.DataFrame(scaler.fit_transform(num_df), columns=num_cols)

    # One hot encoding of cat_df
    encoder = OneHotEncoder()
    cat_df_encoded = pd.DataFrame(encoder.fit_transform(cat_df).toarray(),columns=encoder.get_feature_names_out(cat_cols))

    # Concatenation of the two dataframes
    new_data = pd.concat([num_df_scaled, cat_df_encoded], axis=1)
    return new_data
