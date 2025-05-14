import joblib
import os

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def train_test_split_data(df, target_column="ActualTOW", test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets.

    :param df: pandas DataFrame
    :param target_column: Name of the target column
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random seed for reproducibility
    :return X_train, X_test, y_train, y_test: Split data   
    """
    print("LOG - Splitting data into train and test sets...", end="- ")
    print(f"Target column: {target_column}", end="- ")
    print(f"Test size: {test_size}", end="- ")
    print(f"Random state: {random_state}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_model_pipeline(preprocessor, model):
    """
    Creates a scikit-learn pipeline that includes preprocessing and modeling steps.

    :param preprocessor: sklearn-compatible transformer (e.g., ColumnTransformer)
    :param model: a scikit-learn regressor instance (e.g., LinearRegression())

    :return: A Pipeline object
    """
    print("LOG - Creating model pipeline...")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a regression model. Returns RMSE.

    :param model: Regression model
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training target
    :param y_test: Testing target
    :return rmse: RMSE of the model on the test set
    """
    print("LOG - Evaluating training model ", end="- ")
    print(f"Model: {model.named_steps['regressor']}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse

def prepare_and_save_models(**kwargs):
    """
    Helps training and saves models provided via kwargs.

    :param models: dict of {'model_name': model_object}
    :param output_dir: path to save .pkl files (default: 'models/')
    :param target_column: target column name (default: 'ActualTOW')
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training target
    :param y_test: Testing target
    :return best_rmse: lowest RMSE of the model on the test set
    :return best_model_path: path to the best model saved
    """
    print("LOG - Training and saving models...", end="- ")

    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    missing = [key for key in required_keys if kwargs.get(key) is None]
    if missing:
        raise ValueError(f"Missing required data arguments: {missing}")


    models = kwargs.get("models", {})
    output_dir = kwargs.get("output_dir", "models/")
    target_column = kwargs.get("target_column", "ActualTOW")
    model_name = kwargs.get("model_name", "model.pkl")
    X_train = kwargs.get("X_train")
    X_test = kwargs.get("X_test")   
    y_train = kwargs.get("y_train")
    y_test = kwargs.get("y_test")


    print(f"Output directory: {output_dir}", end="- ")
    print(f"Target column: {target_column}")
    os.makedirs(output_dir, exist_ok=True)

    all_rmses = {}
    all_models = {}

    for name, model in models.items():
        model_name_pkl = f"{name}_{model_name}"
        rmse = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{name} RMSE: {rmse:.2f}")
        
        joblib.dump(model, f"{output_dir}/{model_name_pkl}.pkl")

        all_rmses[name] = rmse
        all_models[name] = model

    return all_rmses, all_models
