import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

import os
import glob

from scripts.preprocessing import handle_missing_values, remove_outliers, build_preprocessor
from scripts.feature_engineering import (
    get_fuel_per_passenger,
    get_total_payload_estimate,
    get_average_baggage_weight,
    feature_engineering_preparation,
    add_datetime_features,
    add_polynomial_features
)
from scripts.training import (
    train_test_split_data,
    prepare_and_save_models,
    create_model_pipeline
)
from scripts.evaluation import evaluate_model

# Configurations
RAW_FILE_PATH = "data/raw/training.csv"
VALIDATION_PATH = "data/raw/validation.csv"
OUTPUT_DIR = "models/"
TARGET_COLUMN = "ActualTOW"
SELECTED_POLY_FEATURES = ["ActualFlightTime", "ActualTotalFuel", "FuelPerPassenger", "AvgBagWeight"]

RMSE_RESULTS = {
    "pipeline": [],
    "missing_strategy": [],
    "outlier_strategy": [],
    "rmse": []
}

# model settings (we have just two models for now), we will add after selecting the best strategy settings
REGRESSORS = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0)
}

# heavier model settings for the best strategy settings
REGRESSORS_BEST_MODEL_SETTINGS = {
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "lasso": Lasso(alpha=0.1, max_iter=5000),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "lasso_cv": LassoCV(cv=5, max_iter=5000),
    "xgboost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# strategy settings
MISSING_STRATEGIES = ["mean", "median", "drop"]
OUTLIER_METHODS = ["iqr", "zscore", None]

DF_RAW = pd.read_csv(RAW_FILE_PATH, sep="\t", na_values=["(null)"])
DF_VAL_RAW = pd.read_csv(VALIDATION_PATH, sep="\t", na_values=["(null)"])

# to not call the same code multiple times, we define functions
def feature_engineering(df):
    """
    Apply feature engineering to the DataFrame.

    :param df: pandas DataFrame
    :return df: DataFrame with new features after feature engineering
    """
    df = feature_engineering_preparation(df)
    df = get_fuel_per_passenger(df)
    df = get_total_payload_estimate(df)
    df = get_average_baggage_weight(df)
    df = add_datetime_features(df)
    df = add_polynomial_features(df, selected_features=SELECTED_POLY_FEATURES, degree=2)

    return df

def handle_missings(df, missing):
    """
    Handle missing values in the DataFrame based on the specified strategy.

    :param df: pandas DataFrame
    :param missing: Strategy for handling missing values ('mean', 'median', 'drop')
    :return df: DataFrame with missing values handled
    """
    if missing != "drop":
        df = handle_missing_values(df, strategy=missing)
    else:
        df = df.dropna()
    return df

def remove_outliers_main(df, outlier):
    """
    Remove outliers from the DataFrame based on the specified method.

    :param df: pandas DataFrame
    :param outlier: Method for handling outliers ('iqr', 'zscore', None)
    :return df: DataFrame with outliers removed
    """
    if outlier:
        df = remove_outliers(df, method=outlier)
    return df

def run_training_pipeline(df, missing, outlier, regressor):
    """
    Run the training pipeline for the given DataFrame and model.

    :param df: pandas DataFrame
    :param missing: Strategy for handling missing values
    :param outlier: Strategy for handling outliers
    :param regressor: Dictionary of regression models
    """
    # third, feature engineering for both train and validation data
    df = feature_engineering(df)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(df, target_column=TARGET_COLUMN)

    # build preprocessor for categorical and numerical features
    preprocessor = build_preprocessor(X_train)

    # prepare model pipelines
    model_pipelines = {
        name: create_model_pipeline(preprocessor, model)
        for name, model in regressor.items()
    }

    # train and save models
    rmses, models = prepare_and_save_models(
        models=model_pipelines,
        output_dir=OUTPUT_DIR,
        target_column=TARGET_COLUMN,
        model_name=f"missing_{missing}_outlier_{outlier or 'none'}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    
    # Store results
    for model_name, pipeline in models.items():
        regressor_name = pipeline.named_steps["regressor"].__class__.__name__
        RMSE_RESULTS["pipeline"].append(regressor_name)
        RMSE_RESULTS["missing_strategy"].append(missing)
        RMSE_RESULTS["outlier_strategy"].append(outlier or "none")
        RMSE_RESULTS["rmse"].append(rmses[model_name])


# loop over strategy combinations but first try on light methods (ridge and linear regression)
def run_basic_models():
    """
    Run basic models (ridge and linear regression) with different missing and outlier strategies.
    """
    for missing in MISSING_STRATEGIES:
        for outlier in OUTLIER_METHODS:
            df = DF_RAW.copy()

            # handle missing values
            df = handle_missings(df, missing)

            # handle outliers
            df = remove_outliers_main(df, outlier)

            run_training_pipeline(df, missing, outlier, REGRESSORS)


# based on the best model settings, we will train the other models which are a bit heavier
def run_advanced_methods(best_setting_missing, best_setting_outlier):
    """
    Run advanced models (lasso, random forest, lasso CV, xgboost) with the best missing and outlier strategies.

    :param best_setting_missing: Best missing strategy
    :param best_setting_outlier: Best outlier strategy
    """
    df = DF_RAW.copy()
    # handle missing values
    df = handle_missings(df, best_setting_missing)

    # handle outliers
    df = remove_outliers_main(df, best_setting_outlier)

    run_training_pipeline(df, best_setting_missing, best_setting_outlier, REGRESSORS_BEST_MODEL_SETTINGS)

def run_evalution(best_model_new_path, best_setting_missing, best_setting_outlier):  
    """
    Evaluate the best model on the validation set.

    :param best_setting_missing: Best missing strategy
    :param best_setting_outlier: Best outlier strategy
    """
    
    df_val = DF_VAL_RAW.copy()
    # handle missing values
    df_val = handle_missings(df_val, best_setting_missing)
    
    # handle outliers
    df_val = remove_outliers_main(df_val, best_setting_outlier)

    # feature engineering
    df_val = feature_engineering(df_val)

    evaluate_model(
    best_model_new_path,
    X_val=df_val
    )

def clear_important_models(best_model_path):
    """
    Clear important models from the output directory by removing all but the best model.

    :param best_model_path: Path to the best model
    """ 
    all_model_files = glob.glob(os.path.join(OUTPUT_DIR, "*.pkl"))

    # Normalize paths for reliable comparison
    best_model_path = os.path.abspath(best_model_path)

    for model_file in all_model_files:
        model_file_abs = os.path.abspath(model_file)

        if model_file_abs != best_model_path:
            os.remove(model_file_abs)
            print(f"LOG - Removed model file: {model_file_abs}")
        else:
            print(f"LOG - Keeping best model file: {model_file_abs}")

if __name__ == "__main__":
    # Run basic models
    run_basic_models()

    # Find the best settings based on RMSE for advanced methods
    best_rmse_index = RMSE_RESULTS["rmse"].index(min(RMSE_RESULTS["rmse"]))
    best_setting_missing = RMSE_RESULTS["missing_strategy"][best_rmse_index]
    best_setting_outlier = RMSE_RESULTS["outlier_strategy"][best_rmse_index]
    
    # Run advanced methods with the best settings
    run_advanced_methods(best_setting_missing, best_setting_outlier)

    # Find the best model based on RMSE
    best_rmse_index = RMSE_RESULTS["rmse"].index(min(RMSE_RESULTS["rmse"]))
    best_model_pipeline = RMSE_RESULTS["pipeline"][best_rmse_index]
    best_model_name = best_model_pipeline.split('(')[0].lower()
    best_model_path = f"{OUTPUT_DIR}/{best_model_name}_missing_{best_setting_missing}_outlier_{best_setting_outlier}.pkl"

    print(f"\n\nLOG - Best model: {best_model_name} - Best model RMSE: {min(RMSE_RESULTS['rmse'])} - \
       Best model missing strategy: {best_setting_missing} - Best model outlier method: {best_setting_outlier}\n\n")
    
    # Save RMSE results to CSV
    RMSE_RESULTS = pd.DataFrame(RMSE_RESULTS)
    RMSE_RESULTS.to_csv("reports/rmse_results.csv", index=False)
    print("LOG - RMSE results saved to reports/rmse_results.csv")

    # Clear all but the best model. If you want to keep all models, comment this out.
    # All the models can be tested on the validation set.
    clear_important_models(best_model_path)
    
    # Evaluate the best model on the validation set
    run_evalution(best_model_path, best_setting_missing, best_setting_outlier)
        

