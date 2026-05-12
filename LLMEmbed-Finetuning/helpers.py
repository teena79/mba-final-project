"""
    This module contains the Helpers class.
"""

# General Imports
import os
import shutil
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
import random
import torch
import pandas as pd

# Local Imports
from config import Config


class Helpers:
    """
    This class contains all the helper methods that are used in this project.
    """

    @classmethod
    def __init__(cls):
        cls.config = Config()

    @classmethod
    def convert_column_to_lowercase(cls, example, column_to_lowercase: str) -> str:
        """
        This method converts the column values into lowercase.
        """
        example[column_to_lowercase] = example[column_to_lowercase].lower()
        return example

    @classmethod
    def replace_string_with_int(cls, example, mapping: dict, column_to_modify: str):
        """
        This function converts string values to integers in a specific column.
        """
        example[column_to_modify] = mapping.get(example[column_to_modify], -1)
        return example

    @classmethod
    def replace_int_with_string(cls, example, mapping: dict, column_to_modify: str):
        """
        This function converts integer values to string in a specific column.
        """
        example[column_to_modify] = mapping.get(example[column_to_modify], "None")
        return example

    @classmethod
    def concatenate_sentimental_analysis_datasets(cls, dataset1, dataset2):
        """
        This method will concatenate the two datasets with same structure.
        """
        print("\n[Started] - Concatenation of the datasets")
        dataset1_df = dataset1["train"].to_pandas()
        dataset2_df = dataset2["train"].to_pandas()
        temp1 = Dataset.from_pandas(dataset1_df)
        temp2 = Dataset.from_pandas(dataset2_df)
        concanted_dataset = concatenate_datasets([temp1, temp2])
        print("[Completed] - Concatenation of the datasets")
        return concanted_dataset

    @classmethod
    def save_embeddings(
        cls, sentences_embeds: list, labels: list, file_path: str, mode: str
    ):
        """
        This method saves the embeddings in a local folder
        """
        save_path = cls.config.get_base_path() + "embeddings/"
        if not os.path.exists(save_path + file_path):
            os.makedirs(save_path + file_path)
        torch.save(
            sentences_embeds.to("cpu"), save_path + file_path + f"{mode}_texts.pt"
        )
        torch.save(labels, save_path + file_path + f"{mode}_labels.pt")
        print(f"Saved at :", save_path + file_path)

    @classmethod
    def save_dataset(cls, dataset, file_name: str):
        """
        This method will save the dataset to a local folder inorder to reuse.
        """
        save_path = cls.config.get_base_path() + "data/"
        file_path = save_path + file_name
        print("\t\tSaved at :", file_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dataset.save_to_disk(file_path)

    @classmethod
    def save_finetuned_model(cls, model, tokenizer, model_name: str):
        """
        This method will save the dataset to a local folder inorder to reuse.
        """
        save_path = cls.config.get_base_path() + "finetuned_models/"
        model_path = save_path + model_name
        print("Saved at :", model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    @classmethod
    def read_dataset_from_local(cls, dataset_name: str):
        """
        This function loads the data from a local directory.
        """
        load_path = cls.config.get_base_path() + "data/" + dataset_name
        if not os.path.exists(load_path):
            raise Exception(
                f"{load_path} does not exists. Please save the data to local again."
            )
            return
        dataset = load_from_disk(load_path)
        return dataset

    @classmethod
    def clear_huggingface_cache(cls):
        """
        This function clears the hugging face cache
        """
        print("\nClearing Hugging Face Cache")
        cache_dir = os.path.expanduser("~/.cache/huggingface/")

        # Check if the directory exists and remove it
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)  # Recursively delete the directory
            print(f"Cache directory: '{cache_dir}' cleared successfully.")
        else:
            print(f"Cache directory: '{cache_dir}' does not exist.")

    @classmethod
    def save_model_results(
        cls, df, finetuned: bool, filename: str, folder: str, task: str
    ):
        """
        This method will save the dataset to a local folder inorder to reuse.
        """
        save_path = cls.config.get_base_path() + "results/"
        base_folder = f"finetuned/{task}/" if finetuned else f"base/{task}/"
        file_path = save_path + base_folder + folder
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(file_path + f"{filename}.csv")
        print("Saved at :", file_path + f"{filename}.csv")

    @classmethod
    def set_seed(cls, seed: int):
        """
        This function sets the seed for all the execution.
        """
        # Set the seed for random number generation in Python
        random.seed(seed)

        # Set the seed for PyTorch on CPU
        torch.manual_seed(seed)

        # Set the seed for PyTorch on all GPUs (if using multiple GPUs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    @classmethod
    def aggregate_results_sentiment_analysis(cls):
        """
        This function aggregates the results for sentiment analysis.
        """
        path = cls.config.get_base_path() + "results"
        save_path = cls.config.get_base_path() + "aggregated_results/"
        task = "sentiment_analysis"
        folders = ["base", "finetuned"]
        tasks, learning_rates, epochs, SIGMA_values = cls.config.get_downstream_model_parameters()
        for folder in folders:
            for mode in cls.config.get_modes():
                epoch_col, lr_col, SIGMA_col, loss, accuracy, micro_f1, macro_f1, avg_loss = [],[],[],[],[],[],[],[]
                index = 0 if mode == "train" else 1
                for epoch in epochs:
                    for lr in learning_rates[task]:
                        for SIGMA in SIGMA_values:
                            SIGMA = round(SIGMA, 1)
                            file_path = f"{path}/{folder}/{task}/EPOCH={epoch}/LR={lr}/SIGMA={SIGMA}/metrics.csv"
                            df = pd.read_csv(file_path)
                            col_1,col_2,col_3,col_4,col_5 = df.loc[index, "loss"],df.loc[index, "accuracy"],df.loc[index, "micro_f1"],df.loc[index, "macro_f1"],df.loc[index, "avg_loss"]
                            epoch_col.append(epoch), lr_col.append(lr), SIGMA_col.append(SIGMA), loss.append(col_1), accuracy.append(col_2), micro_f1.append(col_3), macro_f1.append(col_4), avg_loss.append(col_5)
                # save the files at mode level
                final_df = pd.DataFrame({
                        "epoch": epoch_col,
                        "lr": lr_col,
                        "SIGMA": SIGMA_col,
                        "loss": loss,
                        "accuracy": accuracy,
                        "micro_f1": micro_f1,
                        "macro_f1": macro_f1,
                        "avg_loss": avg_loss
                    })
                final_df.loc[:,"task_type"] = f"{task}_{mode}"
                if not os.path.exists(save_path + folder):
                    os.makedirs(save_path + folder)
                print(f"Read {final_df.shape[0]} files successfully.")
                print(f"Saved at :{save_path}" + f"{folder}/" + f"{task}_{mode}")
                final_df.to_csv(save_path + f"{folder}/" + f"{task}_{mode}.csv")
                del final_df
    
    @classmethod
    def aggregate_results_yes_no_question(cls):
        """
        This function aggregates the results for yes no question.
        """
        path = cls.config.get_base_path() + "results"
        save_path = cls.config.get_base_path() + "aggregated_results/"
        task = "yes_no_question"
        folders = ["base", "finetuned"]
        tasks, learning_rates, epochs, SIGMA_values = cls.config.get_downstream_model_parameters()
        for folder in folders:
            for mode in cls.config.get_modes():
                epoch_col, lr_col, SIGMA_col, loss, accuracy, f1_score = [],[],[],[],[],[]
                index = 0 if mode == "train" else 1
                for epoch in epochs:
                    for lr in learning_rates[task]:
                        for SIGMA in SIGMA_values:
                            SIGMA = round(SIGMA, 1)
                            file_path = f"{path}/{folder}/{task}/EPOCH={epoch}/LR={lr}/SIGMA={SIGMA}/metrics.csv"
                            df = pd.read_csv(file_path)
                            col_1,col_2,col_3= df.loc[index, "loss"],df.loc[index, "accuracy"],df.loc[index, "f1_score"]
                            epoch_col.append(epoch), lr_col.append(lr), SIGMA_col.append(SIGMA), loss.append(col_1), accuracy.append(col_2), f1_score.append(col_3)
                # save the files at mode level
                final_df = pd.DataFrame({
                        "epoch": epoch_col,
                        "lr": lr_col,
                        "SIGMA": SIGMA_col,
                        "loss": loss,
                        "accuracy": accuracy,
                        "f1_score": f1_score,
                    })
                final_df.loc[:,"task_type"] = f"{task}_{mode}"
                if not os.path.exists(save_path + folder):
                    os.makedirs(save_path + folder)
                print(f"Read {final_df.shape[0]} files successfully.")
                print(f"Saved at :{save_path}" + f"{folder}/" + f"{task}_{mode}")
                final_df.to_csv(save_path + f"{folder}/" + f"{task}_{mode}.csv")
                del final_df