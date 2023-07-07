from func.classifier_models import get_model
from func.score_calculation import get_score
import pandas as pd
import logging


def ensemble_classifier(self, train, val, test, models, preds, scores, random_state):
    logging.info("")
    logging.info(f"Ensembling {self.ensemble_ingredients} by {self.ensemble_models}")
    self.progressbar_4.config(maximum=len(self.ensemble_models) + 1, value=0)
    self.progressbar_4.update()
    self.status_label_4.config(text="Predict Probabilities for Ensemble")
    self.status_label_4.update()
    train, val, test = train.copy(), val.copy(), test.copy()

    train_ens = pd.DataFrame({self.target_column: train[self.target_column]})
    val_ens = pd.DataFrame({self.target_column: val[self.target_column]})
    test_ens = pd.DataFrame({self.target_column: test[self.target_column]})

    for ingredient in self.ensemble_ingredients:
        ingredient_model = models[self.syn_model][ingredient]
        train_ens[ingredient] = ingredient_model.predict_proba(train, self.cols)
        val_ens[ingredient] = ingredient_model.predict_proba(val, self.cols)
        test_ens[ingredient] = ingredient_model.predict_proba(test, self.cols)

    ens_cols = {
        "features": self.ensemble_ingredients,
        "current_target": [self.target_column],
    }

    for ens_model in self.ensemble_models:
        logging.info(f"Ensembling by {ens_model}")
        self.progressbar_4.step(1)
        self.progressbar_4.update()
        self.status_label_4.config(text=f"Ensembling by {ens_model}")
        self.status_label_4.update()

        clf_ens = get_model(ens_model, random_state=random_state)
        clf_ens.fit(train_ens, val_ens, ens_cols)
        models[self.syn_model][f"Ensembled by {ens_model}"] = clf_ens
        preds[self.syn_model][f"Ensembled by {ens_model}"] = clf_ens.predict_proba(
            test_ens, ens_cols
        )
        scores[self.syn_model][f"Ensembled by {ens_model}"] = {}
        for metric in self.evaluation_metrics_list:
            scores[self.syn_model][f"Ensembled by {ens_model}"][metric] = get_score(
                test_ens,
                ens_cols,
                preds[self.syn_model][f"Ensembled by {ens_model}"],
                metric,
            )

    self.progressbar_4.config(value=len(self.ensemble_models) + 1)
    self.progressbar_4.update()
    return models, preds, scores
