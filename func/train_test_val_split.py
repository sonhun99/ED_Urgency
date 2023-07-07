from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_val_split(
    df: pd.DataFrame,
    cols: dict,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    # Split the data into train and test
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[cols["current_target"]],
    )

    # Split the train data into train and validation
    train, val = train_test_split(
        train,
        test_size=val_size,
        random_state=random_state,
        stratify=train[cols["current_target"]],
    )

    return train, val, test
