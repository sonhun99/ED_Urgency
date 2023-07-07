from sdv.single_table import (
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
    GaussianCopulaSynthesizer,
)
from sdv.sampling import Condition
from sdv.metadata import SingleTableMetadata
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
import pandas as pd
import pickle
import os
import logging


def get_synthetic_data(
    self,
    df: pd.DataFrame,
    cols: dict,
    model: str = "None",
    random_state: int = 42,
    epochs: int = 300,
):
    def save_models(self, sm):
        # Save the model
        save_path = os.path.join(
            self.output_dir,
            "models",
            f"{self.target_column}_{self.syn_model}_synthesis_model.pkl",
        )
        with open(save_path, "wb") as f:
            pickle.dump(sm, f)

    if model == "None":
        return df
    elif model == "SMOTE":
        categorical_features = df.select_dtypes(["category"]).columns
        if len(categorical_features) > 0:
            sm = SMOTENC(
                categorical_features=categorical_features, random_state=random_state
            )
        else:
            sm = SMOTE(random_state=random_state)
        X, y = sm.fit_resample(df[cols["features"]], df[cols["current_target"]])
        return pd.concat([X, y], axis=1)
    elif model == "ADASYN":
        categorical_features = df.select_dtypes(["category"]).columns
        if len(categorical_features) > 0:
            raise ValueError("ADASYN does not support categorical features")
        sm = ADASYN(random_state=random_state)
        X, y = sm.fit_resample(df[cols["features"]], df[cols["current_target"]])
        return pd.concat([X, y], axis=1)
    elif model in ["CTGAN", "TVAE", "CopulaGAN", "GaussianCopula"]:
        df = df.copy()
        bool_cols = [col for col in df.columns if df[col].dtypes == "bool"]
        df[bool_cols] = df[bool_cols].astype(int)

        if self.load_mode:
            try:
                sm = self.loaded_syn_model
            except:
                sm = None
        else:
            sm = None

        if sm is None:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)

            if model == "CTGAN":
                sm = CTGANSynthesizer(metadata, epochs=epochs)
            elif model == "TVAE":
                sm = TVAESynthesizer(metadata, epochs=epochs)
            elif model == "CopulaGAN":
                sm = CopulaGANSynthesizer(metadata, epochs=epochs)
            elif model == "GaussianCopula":
                # GaussianCopulaSynthesizer does not support epochs
                sm = GaussianCopulaSynthesizer(metadata)
            else:
                raise ValueError(f"Synthesis model {model} not supported")

            sm.fit(df)

        save_models(self, sm)

        conditions = [
            Condition(num_rows=len(df), column_values={cols["current_target"][0]: 1}),
            Condition(num_rows=len(df), column_values={cols["current_target"][0]: 0}),
        ]

        synth_data = sm.sample_from_conditions(conditions=conditions)
        synth_data[bool_cols] = synth_data[bool_cols].astype(bool)
        df[bool_cols] = df[bool_cols].astype(bool)

        # concat synth_data and df
        synth_data = pd.concat([synth_data, df], axis=0)

        return synth_data
    else:
        raise ValueError(f"Synthesis model {model} not supported")
