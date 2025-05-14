import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def get_fuel_per_passenger(df):
    """
    Create domain-specific features based on passenger, fuel, and baggage data.
    This function calculates the fuel per passenger, which can be useful for understanding fuel efficiency.

    :param df: pandas DataFrame
    :return df: DataFrame with fuel per passenger feature
    """
    df = df.copy()

    print("LOG - Creating fuel per passenger feature...")

    # rename columns for clarity
    df.rename(columns={'FLownPassengers': 'FlownPassengers'}, inplace=True)

    # Avoid division by zero
    df["FuelPerPassenger"] = np.where(df["FlownPassengers"] > 0, 
                                  df["ActualTotalFuel"] / df["FlownPassengers"], 
                                  np.nan)

    return df

def get_total_payload_estimate(df):
    """
    Estimate total payload based on passenger count and baggage weight.
    This is a domain-specific feature that can help in understanding the load on the aircraft.

    :param df: pandas DataFrame
    :return df: DataFrame with total payload estimate feature
    """
    df = df.copy()

    print("LOG - Creating total payload estimate feature...")
    # rename columns for clarity
    df.rename(columns={'FLownPassengers': 'FlownPassengers'}, inplace=True)

    df["TotalPayloadEstimate"] = df["FlownPassengers"] * 80 + df["FlightBagsWeight"]
    return df

def get_average_baggage_weight(df):
    """
    Calculate average baggage weight per passenger.
    This is a domain-specific feature that can help in understanding baggage load.

    :param df: pandas DataFrame
    :return df: DataFrame with average baggage weight feature
    """
    df = df.copy()

    print("LOG - Creating average baggage weight feature...")

    df["AvgBagWeight"] = np.where(df["BagsCount"] > 0, 
                                  df["FlightBagsWeight"] / df["BagsCount"], 
                                  np.nan)

    return df

def feature_engineering_preparation(df, numeric_cols = ['ActualFlightTime', 'ActualTotalFuel', 'FLownPassengers', 'BagsCount', 'FlightBagsWeight']):
    """
    Prepares the DataFrame for feature engineering by removing unnecessary columns.

    :param df: pandas DataFrame
    :param numeric_cols: List of numeric columns to convert to numeric types
    :return df: DataFrame with unnecessary columns removed
    """
    print("LOG - Preparing DataFrame for feature engineering...", end="- ")

    df = df.copy()
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def add_polynomial_features(df, selected_features, degree=2):
    """
    Add polynomial features for the specified numeric columns.

    :param df: pandas DataFrame
    :param selected_features: List of columns to expand with polynomial features
    :param degree: Degree of polynomial features to create

    :return df: a DataFrame with original features removed and expanded ones included.
    """

    print("LOG - Creating polynomial features...", end="- ")
    print(f"Degree: {degree}", end="- ")
    print(f"Selected features: {selected_features}")
    df = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    try:
        features = df[selected_features].fillna(0)
    except KeyError as e:
        missing = list(set(selected_features) - set(df.columns))
        raise ValueError(f"Selected features missing in DataFrame: {missing}")

    poly_features = poly.fit_transform(features)
    poly_feature_names = poly.get_feature_names_out(selected_features)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    df.drop(columns=selected_features, inplace=True)
    df = pd.concat([df, df_poly], axis=1)

    return df

def add_datetime_features(df, date_column='DepartureDate'):
    """
    Extracts year, month, day, and weekday from a date column.

    :param df: pandas DataFrame
    :param date_column: Name of the date column to extract features from
    :return df: DataFrame with new features added
    """
    print("LOG - Creating datetime features...", end="- ")
    print(f"Date column: {date_column}")
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    df["DepartureYear"] = df[date_column].dt.year
    df["DepartureMonth"] = df[date_column].dt.month
    df["DepartureDay"] = df[date_column].dt.day
    df["DepartureWeekday"] = df[date_column].dt.weekday

    return df
