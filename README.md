# ED_Urgency

**Scientific Reports (2023)** · [Read the open-access paper](https://www.nature.com/articles/s41598-023-41544-0)

ED_Urgency is the research code for *Improved patient mortality predictions in emergency departments with deep learning data-synthesis and ensemble models*. It compares classifiers and ensembles for predicting in-department mortality from structured emergency department data. The study examines whether resampling and tabular data synthesis help when deaths are much less common than other outcomes.

## Start here

1. [Set up the study environment](#installation).
2. [Prepare a labeled CSV](#input-data).
3. [Launch the application](#run-the-experiment) and select **Start**.
4. [Find the scores and predictions](#outputs) in the timestamped directory under `output/`.

The application is a local Tkinter research interface. It expects a labeled dataset and runs a training and evaluation workflow. The repository contains an empty CSV placeholder, so you need your own authorized data to complete a run.

## Installation

Start with a Linux environment with a graphical display and a CUDA-capable NVIDIA GPU for the default XGBoost configuration. The archived Conda environment specifies Python 3.9.16 and GPU builds of XGBoost and PyTorch. It also pins the versions of LightGBM, scikit-learn, and SDV used by the code.

From a Linux shell:

```bash
git clone https://github.com/sonhun99/ED_Urgency.git
cd ED_Urgency

# Create a local environment file without the archived machine's absolute prefix.
sed '/^prefix:/d' environment.yml > environment.local.yml
conda env create --name ed-urgency --file environment.local.yml
conda activate ed-urgency
```

[environment.yml](environment.yml) records an older, platform-specific setup. The same package builds may be unavailable on other operating systems or newer Conda channels. You can check whether Tkinter can open a window by running `python -m tkinter` in the activated environment.

## Input data

Place an authorized, comma-separated file under `input/`. The launcher chooses the **largest `.csv` file by size** in that directory, so keep only the intended dataset there.

| Requirement | Details |
| --- | --- |
| Column names | Match the names in [parameters.py](parameters.py) exactly, including spaces and capitalization. |
| Predictors | The default configuration expects 21 features, including demographics, clinical measurements, and treatment indicators. |
| Target | The default target is `Death`. Both outcome classes must be present with sufficient examples for a stratified split. |
| Label interpretation | Preprocessing maps two-valued columns to Boolean values in order of first appearance after filtering. Confirm that `Death=True` represents mortality for your input before interpreting probabilities or scores. |

The original study dataset is not publicly distributed. The paper explains the procedure for requesting authorized access. Running the code on a different cohort produces results for that cohort. [Paper and data availability statement](https://www.nature.com/articles/s41598-023-41544-0)

## Run the experiment

From the repository root, with the Conda environment active:

```bash
python main.py
```

In the **ED Urgency Prediction** window, select **Start**. The program runs each configured synthesis and classifier combination and shows its progress in the window. The default experiment includes the original training data as a baseline, oversampling methods such as SMOTE and ADASYN, and SDV synthesizers including CTGAN, TVAE, CopulaGAN, and Gaussian Copula. It evaluates individual classifiers and ensembles that use the individual models' predicted probabilities as inputs.

[parameters.py](parameters.py) lists the input feature names, target, models, and metrics. The default splitting function holds out 20% of the eligible records for testing, then holds out 20% of the remainder for validation. Synthesis and oversampling are applied to the training split. The test split is used to score the trained models.

## Outputs

Each run creates an `output/<timestamp>/` directory. With the default target, look for:

| Output | Purpose |
| --- | --- |
| `Death_scores.csv` | Test-set scores by synthesis method and model, including AUC, F1, precision, recall, accuracy, and IBA (index of balanced accuracy). |
| `predictions/Death_predictions.csv` | Predicted test-set probabilities for the model comparisons. |
| `models/Death_classifier_models.pkl` | Serialized classifier and ensemble objects. |
| `synthesized_data/` | Training tables saved for each completed data synthesis or baseline condition. These can contain original input records. |
| `parameters.py` and `log_<timestamp>.log` | A copy of the configuration and a log of the selected input, progress, and synthesis failures. |

AUC is calculated from model scores. F1, precision, recall, accuracy, and IBA use a probability threshold of 0.5. Review the log alongside the score table, since a synthesis method that fails may be skipped.

## Repository layout

| Path | Role |
| --- | --- |
| [`main.py`](main.py) | Tkinter application and experiment entry point. |
| [`parameters.py`](parameters.py) | Input schema, model lists, and scoring settings. |
| [`environment.yml`](environment.yml) | Archived Conda environment. |
| [`func/`](func/) | Preprocessing, splitting, synthesis, classifiers, ensembles, and metrics. |
| [`input/`](input/) | Location for the local input CSV. |

## Citation

If you use this code in your research, please cite the paper:

Son, B., Myung, J., Shin, Y. et al. Improved patient mortality predictions in emergency departments with deep learning data-synthesis and ensemble models. *Scientific Reports* **13**, 15031 (2023). [https://doi.org/10.1038/s41598-023-41544-0](https://doi.org/10.1038/s41598-023-41544-0)

```bibtex
@article{son2023edurgency,
  author  = {Son, Byounghoon and Myung, Jinwoo and Shin, Younghwan and Kim, Sangdo and Kim, Sung Hyun and Chung, Jong-Moon and Noh, Jiyoung and Cho, Junho and Chung, Hyun Soo},
  title   = {Improved patient mortality predictions in emergency departments with deep learning data-synthesis and ensemble models},
  journal = {Scientific Reports},
  year    = {2023},
  volume  = {13},
  pages   = {15031},
  doi     = {10.1038/s41598-023-41544-0},
  url     = {https://doi.org/10.1038/s41598-023-41544-0}
}
```
