import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import ngboost as ngb
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from pytorch_tabnet import tab_model as tab
import pandas as pd
import numpy as np
import torch


class LGBClassifier:
    def __init__(
        self, num_rounds=1000, early_stopping_rounds=100, verbose=100, random_state=42
    ):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cols: dict):
        self.model = lgb.LGBMClassifier(
            n_estimators=self.num_rounds, random_state=self.random_state
        )
        self.model.fit(
            train_df[cols["features"]],
            train_df[cols["current_target"]],
            eval_set=[(val_df[cols["features"]], val_df[cols["current_target"]])],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
        )

    def predict(self, df: pd.DataFrame, cols: dict):
        return self.model.predict(df[cols["features"]])

    def predict_proba(self, df: pd.DataFrame, cols: dict):
        proba = self.model.predict_proba(df[cols["features"]])
        return proba[:, 1]


class XGBClassifier:
    def __init__(
        self, num_rounds=1000, early_stopping_rounds=100, verbose=100, random_state=42
    ):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cols: dict):
        self.model = xgb.XGBClassifier(
            n_estimators=self.num_rounds,
            random_state=self.random_state,
            enable_categorical=True,
            tree_method="gpu_hist",
        )
        self.model.fit(
            train_df[cols["features"]],
            train_df[cols["current_target"]],
            eval_set=[(val_df[cols["features"]], val_df[cols["current_target"]])],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
        )

    def predict(self, df: pd.DataFrame, cols: dict):
        return self.model.predict(df[cols["features"]])

    def predict_proba(self, df: pd.DataFrame, cols: dict):
        proba = self.model.predict_proba(df[cols["features"]])
        return proba[:, 1]


class CatClassifier:
    def __init__(
        self, num_rounds=1000, early_stopping_rounds=100, verbose=100, random_state=42
    ):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cols: dict):
        self.model = cat.CatBoostClassifier(
            n_estimators=self.num_rounds, random_state=self.random_state
        )
        cat_features = [f for f in train_df.columns if train_df[f].dtype == "category"]
        self.model.fit(
            train_df[cols["features"]],
            train_df[cols["current_target"]],
            eval_set=[(val_df[cols["features"]], val_df[cols["current_target"]])],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            cat_features=cat_features,
        )

    def predict(self, df: pd.DataFrame, cols: dict):
        return self.model.predict(df[cols["features"]])

    def predict_proba(self, df: pd.DataFrame, cols: dict):
        proba = self.model.predict_proba(df[cols["features"]])
        return proba[:, 1]


class NGBClassifier:
    def __init__(
        self, num_rounds=1000, early_stopping_rounds=100, verbose=100, random_state=42
    ):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cols: dict):
        # Since NGBoost does not support categorical features, we need to get rid of it
        cat_features = [f for f in train_df.columns if train_df[f].dtype == "category"]
        # copy train_df and val_df to avoid changing the original data
        train_df = train_df.copy()
        val_df = val_df.copy()

        train_df = train_df.drop(cat_features, axis=1)
        val_df = val_df.drop(cat_features, axis=1)

        # Since NGBoost does not support boolean current_target, we need to make it integer
        train_df[cols["current_target"]] = train_df[cols["current_target"]].astype(int)
        val_df[cols["current_target"]] = val_df[cols["current_target"]].astype(int)

        # cols["features"] also contains categorical features, so we need to remove them
        # but avoid changing the original data
        cols = cols.copy()
        cols["features"] = [f for f in cols["features"] if f not in cat_features]

        self.model = ngb.NGBClassifier(
            n_estimators=self.num_rounds,
            random_state=self.random_state,
            verbose_eval=self.verbose,
        )
        self.model.fit(
            train_df[cols["features"]],
            train_df[cols["current_target"]],
            X_val=val_df[cols["features"]],
            Y_val=val_df[cols["current_target"]],
            early_stopping_rounds=self.early_stopping_rounds,
        )

    def predict(self, df: pd.DataFrame, cols: dict):
        # Since NGBoost does not support categorical features, we need to get rid of it
        cat_features = [f for f in df.columns if df[f].dtype == "category"]
        # copy df to avoid changing the original data
        df = df.copy()
        df = df.drop(cat_features, axis=1)
        # cols["features"] also contains categorical features, so we need to remove them
        # but avoid changing the original data
        cols = cols.copy()
        cols["features"] = [f for f in cols["features"] if f not in cat_features]
        return self.model.predict(df[cols["features"]])

    def predict_proba(self, df: pd.DataFrame, cols: dict):
        # Since NGBoost does not support categorical features, we need to get rid of it
        cat_features = [f for f in df.columns if df[f].dtype == "category"]
        # copy df to avoid changing the original data
        df = df.copy()
        df = df.drop(cat_features, axis=1)
        # cols["features"] also contains categorical features, so we need to remove them
        # but avoid changing the original data
        cols = cols.copy()
        cols["features"] = [f for f in cols["features"] if f not in cat_features]
        proba = self.model.predict_proba(df[cols["features"]])
        return proba[:, 1]


class TabNetClassifier:
    def __init__(
        self, num_rounds=1000, early_stopping_rounds=100, verbose=100, random_state=42
    ):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cols: dict):
        # Since TabNet does not support categorical features, we need to get rid of it
        cat_features = [f for f in train_df.columns if train_df[f].dtype == "category"]
        # copy train_df and val_df to avoid changing the original data
        train_df = train_df.copy()
        val_df = val_df.copy()

        train_df = train_df.drop(cat_features, axis=1)
        val_df = val_df.drop(cat_features, axis=1)

        # cols["features"] also contains categorical features, so we need to remove them
        # but avoid changing the original data
        cols = cols.copy()
        cols["features"] = [f for f in cols["features"] if f not in cat_features]

        self.model = tab.TabNetClassifier(verbose=self.verbose, seed=self.random_state)
        self.model.fit(
            train_df[cols["features"]].to_numpy(dtype=np.float32),
            train_df[cols["current_target"]].to_numpy().flatten(),
            eval_set=[
                (
                    train_df[cols["features"]].to_numpy(dtype=np.float32),
                    train_df[cols["current_target"]].to_numpy().flatten(),
                ),
                (
                    val_df[cols["features"]].to_numpy(dtype=np.float32),
                    val_df[cols["current_target"]].to_numpy().flatten(),
                ),
            ],
            eval_metric=["auc"],
            max_epochs=self.num_rounds,
            patience=self.early_stopping_rounds,
            # batch_size=1024,
            # virtual_batch_size=128,
            # num_workers=0,
            # drop_last=False,
            # loss_fn=torch.nn.functional.cross_entropy,
        )

    def predict(self, df: pd.DataFrame, cols: dict):
        df = df.copy()
        # Since TabNet does not support categorical features, we need to get rid of it
        cat_features = [f for f in df.columns if df[f].dtype == "category"]
        df = df.drop(cat_features, axis=1)
        cols = cols.copy()
        cols["features"] = [f for f in cols["features"] if f not in cat_features]
        return self.model.predict(
            df[cols["features"]].to_numpy(dtype=np.float32)
        ).flatten()

    def predict_proba(self, df: pd.DataFrame, cols: dict):
        df = df.copy()
        # Since TabNet does not support categorical features, we need to get rid of it
        cat_features = [f for f in df.columns if df[f].dtype == "category"]
        df = df.drop(cat_features, axis=1)
        cols = cols.copy()
        cols["features"] = [f for f in cols["features"] if f not in cat_features]
        proba = self.model.predict_proba(
            df[cols["features"]].to_numpy(dtype=np.float32)
        )
        return proba[:, 1]

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)
        return self.model

    def get_feature_importance(self):
        return self.model.feature_importances_


class ScikitClassifier:
    def __init__(
        self,
        num_rounds=1000,
        early_stopping_rounds=100,
        verbose=100,
        random_state=42,
        model_name="LogisticRegression",
    ):
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state
        self.model_name = model_name

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, cols: dict):
        if self.model_name == "AdaBoostClassifier":
            self.model = AdaBoostClassifier(random_state=self.random_state)
        elif self.model_name == "BernoulliNB":
            self.model = BernoulliNB()
        elif self.model_name == "ComplementNB":
            self.model = ComplementNB()
        elif self.model_name == "DecisionTreeClassifier":
            self.model = DecisionTreeClassifier(random_state=self.random_state)
        elif self.model_name == "ExtraTreeClassifier":
            self.model = ExtraTreeClassifier(random_state=self.random_state)
        elif self.model_name == "GaussianNB":
            self.model = GaussianNB()
        elif self.model_name == "GaussianProcessClassifier":
            self.model = GaussianProcessClassifier(random_state=self.random_state)
        elif self.model_name == "GradientBoostingClassifier":
            self.model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_name == "HistGradientBoostingClassifier":
            self.model = HistGradientBoostingClassifier(random_state=self.random_state)
        elif self.model_name == "KNeighborsClassifier":
            self.model = KNeighborsClassifier()
        elif self.model_name == "LinearDiscriminantAnalysis":
            self.model = LinearDiscriminantAnalysis()
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression(random_state=self.random_state)
        elif self.model_name == "MLPClassifier":
            self.model = MLPClassifier(random_state=self.random_state)
        elif self.model_name == "QuadraticDiscriminantAnalysis":
            self.model = QuadraticDiscriminantAnalysis()
        elif self.model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(random_state=self.random_state)
        elif self.model_name == "SGDClassifier":
            self.model = SGDClassifier(loss="log", random_state=self.random_state)
        elif self.model_name == "SVC":
            self.model = SVC(probability=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        self.model.fit(
            train_df[cols["features"]],
            train_df[cols["current_target"]].values.ravel(),
        )

    def predict(self, df: pd.DataFrame, cols: dict):
        df = df.copy()
        return self.model.predict(df[cols["features"]]).flatten()

    def predict_proba(self, df: pd.DataFrame, cols: dict):
        df = df.copy()
        proba = self.model.predict_proba(df[cols["features"]])
        return proba[:, 1]


def get_model(model_name: str, **kwargs):
    scikitclassifiers = [
        "AdaBoostClassifier",
        # "BaggingClassifier",
        "BernoulliNB",
        # "CalibratedClassifierCV",
        "ComplementNB",
        "DecisionTreeClassifier",
        "ExtraTreeClassifier",
        "GaussianNB",
        "GaussianProcessClassifier",
        "GradientBoostingClassifier",
        # "GridSearchCV",
        # "HalvingGridSearchCV",
        # "HalvingRandomSearchCV",
        "HistGradientBoostingClassifier",
        "KNeighborsClassifier",
        "LinearDiscriminantAnalysis",
        "LogisticRegression",
        # "LogisticRegressionCV",
        "MLPClassifier",
        "QuadraticDiscriminantAnalysis",
        "RandomForestClassifier",
        # "RandomizedSearchCV",
        "SGDClassifier",
        "SVC",
        # "StackingClassifier",
        # "VotingClassifier",
    ]

    if model_name == "LightGBM":
        return LGBClassifier(**kwargs)
    elif model_name == "XGBoost":
        return XGBClassifier(**kwargs)
    elif model_name == "CatBoost":
        return CatClassifier(**kwargs)
    elif model_name == "NGBoost":
        return NGBClassifier(**kwargs)
    elif model_name == "TabNet":
        return TabNetClassifier(**kwargs)
    elif model_name in scikitclassifiers:
        return ScikitClassifier(**kwargs, model_name=model_name)
    else:
        raise ValueError(f"Claasifier model {model_name} is not supported.")
