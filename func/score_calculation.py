from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import pandas as pd


def iba_score(y_true, y_pred, alpha=0.1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= 0.5).ravel()
    tpr, tnr = tp / (tp + fn), tn / (tn + fp)
    return (1 + alpha * (tpr - tnr)) * tpr * tnr


def get_score(df: pd.DataFrame, cols: dict, pred, metric: str):
    if metric == "AUC":
        return roc_auc_score(df[cols["current_target"]], pred)
    elif metric == "F1":
        return f1_score(df[cols["current_target"]], pred >= 0.5)
    elif metric == "Accuracy":
        return accuracy_score(df[cols["current_target"]], pred >= 0.5)
    elif metric == "Precision":
        return precision_score(df[cols["current_target"]], pred >= 0.5)
    elif metric == "Recall":
        return recall_score(df[cols["current_target"]], pred >= 0.5)
    elif metric == "IBA":
        return iba_score(df[cols["current_target"]], pred)
    else:
        raise ValueError(f"Metric {metric} is not supported")
