import os
import logging
import time
import shutil
import pickle
from datetime import timedelta
from datetime import datetime
import pandas as pd
from tkinter import *
from tkinter.ttk import *
from func.preprocessing import preprocess
from func.train_test_val_split import train_test_val_split
from func.classifier_models import get_model
from func.synthesis_models import get_synthetic_data
from func.score_calculation import get_score
from func.ensemble import ensemble_classifier
from parameters import *


class App(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.configure_logging()
        logging.info("App started")
        self.configure_etc()
        self.create_widgets()

    def configure_logging(self):
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create a file handler for the log file
        self.init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("output", self.init_time)
        os.makedirs(self.output_dir, exist_ok=True)
        log_filename = f"log_{self.init_time}.log"
        file_handler = logging.FileHandler(os.path.join(self.output_dir, log_filename))
        file_handler.setLevel(logging.INFO)

        # Create a formatter for the log messages
        log_format = "%(asctime)s [%(levelname)s] %(message)s"
        formatter = logging.Formatter(log_format)

        # Add the formatter to the file handler
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    def configure_etc(self):
        # find the input file
        self.input_file = self.find_input_file()

        # check if parameters.py exists
        if not os.path.exists("parameters.py"):
            logging.error("parameters.py not found")
            raise Exception("parameters.py not found")

        # copy and paste parameters.py to output directory
        shutil.copyfile("parameters.py", os.path.join(self.output_dir, "parameters.py"))

        self.cols = columns.copy()
        self.ensemble_ingredients = ensemble_ingredients.copy()
        self.evaluation_metrics_list = evaluation_metrics_list.copy()
        self.ensemble_models = ensemble_models.copy()

        os.makedirs(os.path.join(self.output_dir, "synthesized_data"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "predictions"), exist_ok=True)

        try:
            # Load the models
            self.load_dir = load_dir
            logging.info(f"Load Models from {load_dir}")
            self.load_mode = True
        except:
            self.load_dir = None
            logging.info("No pre-trained models found")
            self.load_mode = False

    def find_input_file(self):
        input_file_list = [
            file for file in os.listdir("input") if file.endswith(".csv")
        ]

        if not input_file_list:
            logging.error("No input file found")
            raise Exception("No input file found")

        self.input_file = max(
            (os.path.join("input", f) for f in input_file_list), key=os.path.getsize
        )

        logging.info(f"Input file found: {self.input_file}")
        return self.input_file

    def create_widgets(self):
        # Create progress bars
        self.progressbar_1 = Progressbar(self, length=400, mode="determinate")
        self.progressbar_2 = Progressbar(self, length=400, mode="determinate")
        self.progressbar_3 = Progressbar(self, length=400, mode="determinate")
        self.progressbar_4 = Progressbar(self, length=400, mode="determinate")

        # Create status labels
        self.status_label_1 = Label(self, text="")
        self.status_label_2 = Label(self, text="")
        self.status_label_3 = Label(self, text="")
        self.status_label_4 = Label(self, text="")
        self.status_label_5 = Label(self, text="Ready")

        # Create start button
        self.start_button = Button(self, text="Start", command=self.start_task)

        # Pack the widgets
        self.progressbar_1.pack(pady=10)
        self.status_label_1.pack()
        self.progressbar_2.pack(pady=10)
        self.status_label_2.pack()
        self.progressbar_3.pack(pady=10)
        self.status_label_3.pack()
        self.progressbar_4.pack(pady=10)
        self.status_label_4.pack()
        self.status_label_5.pack(pady=10)
        self.start_button.pack(pady=10)

    def format_time(self, seconds):
        return str(timedelta(seconds=seconds))

    def save_etc(self, models, preds, scores, target_column):
        # Save scores to CSV
        scores_df = pd.DataFrame.from_dict(
            {(i, j): scores[i][j] for i in scores.keys() for j in scores[i].keys()},
            orient="index",
        )
        scores_df.to_csv(os.path.join(self.output_dir, f"{target_column}_scores.csv"))

        # Save predictions to CSV
        preds_df = pd.DataFrame.from_dict(
            {(i, j): preds[i][j] for i in preds.keys() for j in preds[i].keys()},
        )
        preds_df.to_csv(
            os.path.join(
                self.output_dir, "predictions", f"{target_column}_predictions.csv"
            )
        )

        # Save predictions with pickle
        with open(
            os.path.join(
                self.output_dir, "predictions", f"{target_column}_predictions.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(preds, f)

        # Save models to pickle
        with open(
            os.path.join(
                self.output_dir, "models", f"{target_column}_classifier_models.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(models, f)

    def start_task(self):
        # start log
        logging.info("Task started")

        # Record the start time
        start_time = time.time()

        # Disable the start button
        self.start_button.config(state=DISABLED, text="Running...")
        self.status_label_5.config(text="")

        # Iterate over each input file
        for idx_1, target_column in enumerate(self.cols["targets"]):
            # Record the start time
            start_time_1 = time.time()

            # Update the progress bar and status label
            self.progressbar_1.config(maximum=len(self.cols["targets"]), value=idx_1)
            self.status_label_1.config(
                text=f"Target Column : {target_column} , {idx_1}/{len(self.cols['targets'])}"
            )
            self.update()

            # Log the current target column
            logging.info(
                f"Target Column : {target_column} , {idx_1}/{len(self.cols['targets'])}"
            )

            # Update the current target column
            self.cols["current_target"] = [target_column]
            self.target_column = target_column

            # Read the raw data from the input file
            raw_df = pd.read_csv(self.input_file)

            # Preprocess the data
            logging.info("Preprocessing the data")
            df = preprocess(raw_df, self.cols)

            # Split the data into train, validation, and test sets
            logging.info("Splitting the data into train, validation, and test sets")
            logging.info("")
            train, val, test = train_test_val_split(df, self.cols)

            models, preds, scores = {}, {}, {}

            # Iterate over each synthesizer model
            for idx_2, syn_model in enumerate(synthesizer_models_list):
                # Record the start time
                start_time_2 = time.time()

                # Log the current synthesizer model
                logging.info(
                    f"Synthesizer Model : {syn_model} , {idx_2}/{len(synthesizer_models_list)}"
                )

                # Update progress bar and status label with the current synthesizer model
                self.progressbar_2.config(
                    maximum=len(synthesizer_models_list), value=idx_2
                )
                self.status_label_2.config(
                    text=f"Synthesizer Model : {syn_model} , {idx_2}/{len(synthesizer_models_list)}"
                )
                self.status_label_3.config(text="")
                self.progressbar_3.config(value=0)
                self.status_label_4.config(text="")
                self.progressbar_4.config(value=0)
                self.update()

                self.syn_model = syn_model

                models[syn_model], preds[syn_model], scores[syn_model] = {}, {}, {}

                if self.load_mode:
                    try:
                        load_path = os.path.join(
                            self.load_dir,
                            "models",
                            f"{self.target_column}_{self.syn_model}_synthesis_model.pkl",
                        )
                        with open(load_path, "rb") as f:
                            self.loaded_syn_model = pickle.load(f)
                    except:
                        self.loaded_syn_model = None
                else:
                    self.loaded_syn_model = None

                # Generate synthetic data using the current synthesizer model
                start_time_2_1 = time.time()
                try:
                    train_synth = get_synthetic_data(
                        self,
                        train,
                        self.cols,
                        syn_model,
                        RANDOM_STATE,
                        NUM_ROUNDS_FOR_SDV,
                    )
                except:
                    logging.info(f"Synthesizer model {syn_model} failed")
                    continue
                end_time_2_1 = time.time()
                logging.info(
                    f"Execution time for synthesizer {syn_model} : {self.format_time(end_time_2_1 - start_time_2_1)}"
                )

                # Save the synthetic data
                syn_path = os.path.join(
                    self.output_dir,
                    "synthesized_data",
                    f"synthesized_data - {target_column} - {syn_model}.csv",
                )
                train_synth.to_csv(syn_path, index=False)

                if self.load_mode:
                    try:
                        load_path = os.path.join(
                            self.load_dir,
                            "models",
                            f"{target_column}_classifier_models.pkl",
                        )
                        with open(load_path, "rb") as f:
                            loaded_models = pickle.load(f)
                    except:
                        loaded_models = None

                # Iterate over each classifier model
                for idx_3, clf_model in enumerate(classifier_models_list):
                    # Record the start time
                    start_time_3 = time.time()

                    # Log the current classifier model
                    logging.info(
                        f"Classifier Model : {clf_model} , {idx_3}/{len(classifier_models_list)}"
                    )

                    # Update progress bar and status label with the current classifier model
                    self.progressbar_3.config(
                        maximum=len(classifier_models_list), value=idx_3
                    )
                    self.status_label_3.config(
                        text=f"Classifier Model : {clf_model} , {idx_3}/{len(classifier_models_list)}"
                    )
                    self.update()

                    clf = None
                    try:
                        clf = loaded_models[syn_model][clf_model]
                        # logging.info("Load model from ", load_path)
                        logging.info(
                            f"Loaded {syn_model} - {clf_model} classifier model"
                        )
                    except:
                        clf = None

                    # Train the classifier model using the synthetic data
                    if clf is None:
                        clf = get_model(
                            clf_model,
                            num_rounds=NUM_ROUNDS_FOR_CLASSIFIER,
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            verbose=VERBOSE,
                            random_state=RANDOM_STATE,
                        )
                        start_time_3_1 = time.time()
                        clf.fit(train_synth, val, self.cols)
                        end_time_3_1 = time.time()
                        logging.info(
                            f"Execution time for training classifier {clf_model} : {self.format_time(end_time_3_1 - start_time_3_1)}"
                        )

                    # Save the trained model
                    models[syn_model][clf_model] = clf

                    # Predict on the test set
                    preds[syn_model][clf_model] = clf.predict_proba(test, self.cols)

                    scores[syn_model][clf_model] = {}
                    # Calculate scores for evaluation metrics
                    for metric in evaluation_metrics_list:
                        scores[syn_model][clf_model][metric] = get_score(
                            test, self.cols, preds[syn_model][clf_model], metric
                        )

                    self.progressbar_3.config(value=idx_3 + 1)

                    # Record the end time
                    end_time_3 = time.time()

                    # save log execution time
                    logging.info(
                        f"Target : {target_column} - Synthesizer : {syn_model} - Classifier : {clf_model} Execution Time : {self.format_time(end_time_3 - start_time_3)}"
                    )

                # Ensemble the classifier models
                models, preds, scores = ensemble_classifier(
                    self, train_synth, val, test, models, preds, scores, RANDOM_STATE
                )

                # Update progress bar and status label with the current synthesizer model
                self.progressbar_2.config(value=idx_2 + 1)

                # Record the end time
                end_time_2 = time.time()

                # log execution time
                logging.info(
                    f"Target : {target_column} - Synthesizer : {syn_model} Execution Time : {self.format_time(end_time_2 - start_time_2)}"
                )
                logging.info("")

            # Save the trained models
            self.save_etc(models, preds, scores, target_column)

            # Update progress bar and status label with the current target column
            self.progressbar_1.config(value=idx_1 + 1)

            # Record the end time
            end_time_1 = time.time()

            # save log execution time
            logging.info(
                f"{target_column} Execution Time : {self.format_time(end_time_1 - start_time_1)}"
            )
            logging.info("")

        # make the button clickable again
        self.start_button.config(state="normal", text="Close")
        # when it is clicked, close the program
        self.start_button.config(command=self.master.destroy)

        # Record the end time
        end_time = time.time()

        # Update progress bar and status label with the current target column
        self.status_label_5.config(
            text=f"Total Execution Time : {self.format_time(end_time - start_time)}"
        )
        logging.info(
            f"Total Execution Time : {self.format_time(end_time - start_time)}"
        )


if __name__ == "__main__":
    root = Tk()
    root.title("ED Urgency Prediction")
    app = App(master=root)
    app.mainloop()

