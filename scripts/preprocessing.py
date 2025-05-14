import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import zscore

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# helper func
def get_numeric_columns(df, columns):
    """
    Get numeric columns from the DataFrame. If columns are not specified, return all numeric columns.

    :param df: pandas DataFrame
    :param columns: List of columns to check. If None, returns all numeric columns.
    :return columns: List of numeric column names
    """
    if columns is None:
        return df.select_dtypes(include=[np.number]).columns.tolist()
    return columns


def handle_missing_values(df, strategy='median', columns=None):
    """
    Handle missing values in the specified columns using the given strategy for both categorical and numerical values.
    
    :param df: pandas DataFrame
    :param strategy: 'mean', 'median', 'most_frequent', or 'constant'
    :param columns: List of columns to apply imputation. If None, applies to all numeric columns.

    :return df: with missing values handled
    """
    df_copy = df.copy()
    print("LOG - Handling missing values...", end="- ")
    print(f"Strategy: {strategy}", end="- ")
    print(f"Columns: {columns}")
    
    columns = get_numeric_columns(df_copy, columns)

    imputer = SimpleImputer(strategy=strategy)
    df_copy[columns] = imputer.fit_transform(df_copy[columns])

    return df_copy


def winsorize_columns(df, columns=None, limits=(0.01, 0.01)):
    """
    Applies winsorization to numeric columns to limit extreme values.

    :param df: pandas DataFrame
    :param columns: List of columns to winsorize. If None, applies to all numeric columns.
    :param limits: tuple specifying the percentile limits for winsorization

    :return df: with winsorized columns
    """
    df_copy = df.copy()

    print("LOG - Winsorizing columns...", end="- ")
    print(f"Limits: {limits}", end="- ")
    print(f"Columns: {columns}")

    columns = get_numeric_columns(df_copy, columns)

    for col in columns:
        df_copy[col] = np.array(winsorize(df_copy[col], limits=limits))

    return df_copy


def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Removes rows containing outliers based on the specified method.

    :param df: pandas DataFrame
    :param columns: List of columns to check for outliers. If None, applies to all numeric columns.
    :param method: Method to detect outliers ('iqr' or 'zscore').
    :param threshold: Threshold value for defining outliers (IQR multiplier or Z-score threshold)

    :return df: with outliers removed
    """
    df_copy = df.copy()

    print("LOG - Removing outliers...", end="- ")
    print(f"Method: {method}", end="- ")
    print(f"Threshold: {threshold}", end="- ")
    print(f"Columns: {columns}")

    columns = get_numeric_columns(df_copy, columns)

    if method == 'iqr':
        for col in columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]

    elif method == 'zscore':
        z_scores = np.abs(zscore(df_copy[columns]))
        filter_entries = ~((z_scores > threshold).any(axis=1))
        df_copy = df_copy.loc[filter_entries]

    return df_copy

def build_preprocessor(X):
    """
    Constructs a ColumnTransformer with:
    - numeric pipeline: imputation + scaling
    - categorical pipeline: imputation + one-hot encoding

    :param X: The input features (not including target)

    :return preprocessor: ColumnTransformer object with pipelines for numeric and categorical features
    """
    print("LOG - Building preprocessor...", end="- ")
    print(f"Input shape: {X.shape}")

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    return preprocessor
