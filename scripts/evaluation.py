import joblib
import pandas as pd

def evaluate_model(best_model, X_val):
    """
    Loads a pre-trained model from disk and evaluates it on validation data and saves them to a csv file.

    :param best_model: Path to the best model file
    :param X_val: Validation features
    """

    print(f"LOG - Evaluating model")
    save_path = 'reports/predictions.csv'

    model = joblib.load(best_model)
    y_pred = model.predict(X_val)
    # save the results to a CSV file
    results_df = pd.DataFrame({
        'Predicted': y_pred
    })
    results_df.to_csv(save_path, index=False)
    print(f"LOG - Predictions saved to {save_path}")



