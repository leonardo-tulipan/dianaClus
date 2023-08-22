import pandas as pd
import numpy as np
from joblib import load
from warnings import filterwarnings
from sklearn.preprocessing import PowerTransformer




def cluster_counts(df, column):
    counts = df[column].value_counts()
    counts = counts.reset_index()
    counts.columns = ['LABELS', 'COUNT']
    counts = counts.sort_values(['LABELS'])
    return counts


def cluster_coverage(df, column):
    clustered = df[column] >= 0
    coverage = np.sum(clustered) / len(df)
    return clustered, coverage


def load_clf(path):
    """
    Given a pretrained DenseClus this method loads the model from path
    :param path: place where the model is stored
    :return: embedding with labels and clf
    """
    clf = load(path)
    labels = clf.score()
    embedding = pd.DataFrame(clf.mapper_.embedding_)
    embedding['LABELS'] = labels

    return embedding, clf


def check_is_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Requires DataFrame as input")


def extract_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts categorical features into binary dummy dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: binary dummy DataFrame of categorical features
    """
    check_is_df(df)

    categorical = df.select_dtypes(exclude=["float", "int"])
    if categorical.shape[1] == 0:
        raise ValueError("No Categories found, check that objects are in dataframe")

    categorical_dummies = pd.get_dummies(categorical)

    return categorical_dummies


def extract_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numerical features into normailzed numeric only dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: normalized numerical DataFrame of numerical features
    """
    check_is_df(df)

    numerical = df.select_dtypes(include=["float", "int"])
    if numerical.shape[1] == 0:
        raise ValueError("No numerics found, check that numerics are in dataframe")

    return transform_numerics(numerical)


def transform_numerics(numerical: pd.DataFrame) -> pd.DataFrame:
    """Power transforms numerical DataFrame

    Parameters:
        numerical (pd.DataFrame): Numerical features DataFrame

    Returns:
        pd.DataFrame: Normalized DataFrame of Numerical features
    """

    check_is_df(numerical)

    for names in numerical.columns.tolist():
        pt = PowerTransformer(copy=False)
        # TO DO: fix this warning message
        filterwarnings("ignore")
        numerical.loc[:, names] = pt.fit_transform(
            np.array(numerical.loc[:, names]).reshape(-1, 1),
        )
        filterwarnings("default")

    return numerical


def normalize_array(arr):
    # Calculate the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array using the formula: (x - min) / (max - min)
    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr, min_val, max_val


def normalize_new(arr, min_val, max_val):
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
