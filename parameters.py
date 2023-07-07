import os

RANDOM_STATE = 42  # default: 42
TRAIN_TEST_RATIO = 0.2  # default: 0.2
VERBOSE = 100  # default: 100
EARLY_STOPPING_ROUNDS = 500  # default: 500
NUM_ROUNDS_FOR_CLASSIFIER = 5000  # default: 5000
NUM_ROUNDS_FOR_SDV = 300  # default: 300
# load_dir = None  # default : None
columns = {
    "features": [
        "Sex",
        "Age",
        "O2",
        "Nebulizer",
        "Chest X-ray",
        "CBC",
        "Fluid",
        "Discharge med.",
        "HiBP",
        "DM",
        "Pul.Tbc",
        "Allergy",
        "Hepatitis",
        "Other medication",
        "mental status",
        "SBP",
        "DBP",
        "BT",
        "O2Sat",
        "PR",
        "RR",
    ],
    "targets": [
        "Death",
        "Final Patient Status",
        "ICU transfer",
    ],
}
classifier_models_list = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "NGBoost",
    "TabNet",
    "AdaBoostClassifier",
    "BernoulliNB",
    "ComplementNB",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "GaussianNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "KNeighborsClassifier",
    "LinearDiscriminantAnalysis",
    "LogisticRegression",
    "MLPClassifier",
    "QuadraticDiscriminantAnalysis",
    "RandomForestClassifier",
    "SGDClassifier",
    "SVC",
]
synthesizer_models_list = [
    "None",
    "SMOTE",
    "ADASYN",
    "CTGAN",
    "TVAE",
    "CopulaGAN",
    "GaussianCopula",
]
evaluation_metrics_list = [
    "AUC",
    "IBA",
    "F1",
    "Accuracy",
    "Precision",
    "Recall",
]
ensemble_ingredients = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "NGBoost",
    "TabNet",
    "AdaBoostClassifier",
    "BernoulliNB",
    "ComplementNB",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "GaussianNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "KNeighborsClassifier",
    "LinearDiscriminantAnalysis",
    "LogisticRegression",
    "MLPClassifier",
    "QuadraticDiscriminantAnalysis",
    "RandomForestClassifier",
    "SGDClassifier",
    "SVC",
]
ensemble_models = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "LogisticRegression",
    "RandomForestClassifier",
]
