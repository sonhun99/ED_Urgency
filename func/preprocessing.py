import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def preprocess(
    df: pd.DataFrame, cols: dict, impute: str = "None", scale: str = "None"
) -> pd.DataFrame:
    # Make a copy of the dataframe
    df = df.copy()

    # Make sure that the columns are in the dataframe
    for col in cols["features"] + cols["current_target"]:
        if col not in df.columns:
            raise ValueError(f"Column {col} is not in the dataframe")

    # Drop columns which are not features or current_target
    unused_cols = []
    for col in df.columns:
        if col not in cols["features"] + cols["current_target"]:
            unused_cols.append(col)
            logging.info(f"Column {col} is not used")
    df.drop(columns=unused_cols, inplace=True)

    # Drop rows with null current_target
    # Count how many rows have null current_target
    null_targets = df[cols["current_target"]].isnull().sum()
    if null_targets.values > 0:
        df.dropna(subset=cols["current_target"], inplace=True)
        logging.info(f"Drop {null_targets} rows with null current_target")

    # Impute missing values
    if impute == "mean":
        imputer = SimpleImputer(strategy="mean")
        df[cols["features"]] = imputer.fit_transform(df[cols["features"]])
    elif impute == "median":
        imputer = SimpleImputer(strategy="median")
        df[cols["features"]] = imputer.fit_transform(df[cols["features"]])
    elif impute == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
        df[cols["features"]] = imputer.fit_transform(df[cols["features"]])
    elif impute == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        df[cols["features"]] = imputer.fit_transform(df[cols["features"]])
    elif impute == "None":
        pass
    else:
        raise ValueError("Invalid impute method")

    # Drop rows with null feautures
    # Count how many rows have null features
    null_features = df[cols["features"]].isnull().sum().sum()
    if null_features > 0:
        df.dropna(subset=cols["features"], inplace=True)
        logging.info(f"Drop {null_features} rows with null features")

    # Find the columns that are not useful
    # Drop the columns that are not useful
    useless_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            useless_cols.append(col)
            logging.info(f"Column {col} is useless")
    df.drop(columns=useless_cols, inplace=True)

    # Find boolean columns and convert them to bool type
    bool_cols = []
    for col in df.columns:
        if df[col].nunique() == 2:
            bool_cols.append(col)
            logging.info(f"Column {col} is boolean")
            df[col] = df[col].map(
                {df[col].unique()[0]: False, df[col].unique()[1]: True}
            )
    df[bool_cols] = df[bool_cols].astype("bool")

    # Find the columns that are categorical
    # Make categorical columns into category type
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            categorical_cols.append(col)
            logging.info(f"Column {col} is categorical")
    df[categorical_cols] = df[categorical_cols].astype("category")

    # Scale the features
    if scale == "standard":
        scaler = StandardScaler()
        df[cols["features"]] = scaler.fit_transform(df[cols["features"]])
    elif scale == "minmax":
        scaler = MinMaxScaler()
        df[cols["features"]] = scaler.fit_transform(df[cols["features"]])
    elif scale == "robust":
        scaler = RobustScaler()
        df[cols["features"]] = scaler.fit_transform(df[cols["features"]])
    elif scale == "None":
        pass
    else:
        raise ValueError("Invalid scale method")

    # show what we have done
    logging.info(df.info())

    # show what is inside the current_target column
    logging.info(df[cols["current_target"]].value_counts())
    return df
