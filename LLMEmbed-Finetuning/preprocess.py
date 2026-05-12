"""
    This module contains the Preprocess class.
"""

# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger

# General Imports
from datasets import load_dataset, Dataset
import warnings
import numpy as np
import pandas as pd
import re

warnings.filterwarnings("ignore")


class Preprocess:
    """
    This class performs the following tasks :
        1. Load the financial_phrasebank and Sujet-Finance-Instruct-177k from hugging face.
        2. Filter out the selected task types from the Sujet-Finance-Instruct-177k dataset.
        3. Convert the labels, texts into lowercase in Sujet-Finance-Instruct-177k dataset.
        4. Rename column names in Sujet-Finance-Instruct-177k dataset.
        5. Seggregate the Sujet-Finance-Instruct-177k dataset according to the task types.
        6. Check for the null rows and drop them if required.
        7. Check for the duplicate rows and drop them if required.
        8. Peform preprocessing of yes no question data, i.e, exploding the rows.
        9. Create seperate datasets for fine tuning and normal flow.
        10. Replace the string labels with integers in Sujet-Finance-Instruct-177k dataset.
        11. Return the final required datasets.
    """

    @classmethod
    def __init__(cls, enable_logging):
        """
        This method initializes the dictionary to save the datasets.
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.log = Logger()
        cls.temp_datasets = {}
        cls.enable_logging = enable_logging
        cls.fine_tune_split_size = 0.30
        cls.seed = 0

    @classmethod
    def load_datasets(cls) -> dict:
        """
        This method will load the datasets and save in a dictionary as key value pair.
        """
        datasets = {}
        datasets["sujet_finance"] = load_dataset(
            "sujet-ai/Sujet-Finance-Instruct-177k"
        )["train"]
        return datasets

    @classmethod
    def convert_to_pandas_df(cls) -> dict:
        """
        This method will convert the hugging face datasets into pandas dataframe for better visualization.
        """
        df = {}
        df["sujet_finance"] = cls.temp_datasets["sujet_finance"].to_pandas()
        return df

    @classmethod
    def select_task_types_columns(cls, dataset) -> dict:
        """
        This method will filter out the task types, and also select only few columns from the sujet_finance dataset.
        The selected task types : ["sentiment_analysis","yes_no"]
        """
        selected_task_types = cls.config.get_selected_task_types()
        selected_columns = cls.config.get_selected_columns()
        dataset = dataset.filter(lambda x: x["task_type"] in selected_task_types)
        dataset = dataset.select_columns(selected_columns)
        return dataset

    @classmethod
    def convert_lower_case(cls, datasets) -> dict:
        """
        This method will convert the column values to lowercase in the datasets.
        """
        datasets["sujet_finance"] = datasets["sujet_finance"].map(
            cls.helpers.convert_column_to_lowercase,
            fn_kwargs={"column_to_lowercase": "answer"},
        )
        return datasets

    @classmethod
    def rename_column_names(cls, datasets) -> dict:
        """
        This method will rename the columns names.
        """
        rename_map_dict = cls.config.get_rename_column_names_mapping()
        for old_name, new_name in rename_map_dict["sujet"].items():
            datasets["sujet_finance"] = datasets["sujet_finance"].rename_column(
                old_name, new_name
            )
        return datasets

    @classmethod
    def seggregate_sujet_task_types(cls, datasets) -> dict:
        """
        This method seggregates the datasets based on the task types and order the columns in the sujet_finance datasets.
        """
        task_types = cls.config.get_selected_task_types()
        columns_order = cls.config.get_columns_order()
        for task_type in task_types:
            datasets[f"sujet_finance_{task_type}"] = (
                datasets["sujet_finance"]
                .filter(lambda x: x["task_type"] == task_type)
                .select_columns(columns_order)
            )
        return datasets

    @classmethod
    def drop_null_rows(cls, datasets) -> dict:
        """
        This method will drop the null rows in the datasets.
        """
        for dataset_name, dataset in datasets.items():
            cls.log.log(
                message=f"\n\t[Started] - Null rows removal for {dataset_name}.",
                enable_logging=cls.enable_logging,
            )
            column_names = dataset.column_names
            cls.log.log(
                message=f"\t\tNo.of rows in {dataset_name} : {len(dataset)}.",
                enable_logging=cls.enable_logging,
            )
            filtered_dataset = dataset.filter(
                lambda example: all(
                    example[col] is not None
                    and not (isinstance(example[col], float) and np.isnan(example[col]))
                    for col in column_names
                )
            )
            no_of_null_rows = len(dataset) - len(filtered_dataset)
            cls.log.log(
                message=f"\t\tRemoved {no_of_null_rows} null rows in {dataset_name}.",
                enable_logging=cls.enable_logging,
            )
            datasets[dataset_name] = filtered_dataset
            cls.log.log(
                message=f"\t[Completed] - Null rows removal for {dataset_name}.",
                enable_logging=cls.enable_logging,
            )
        return datasets

    @classmethod
    def drop_duplicate_rows(cls, datasets) -> dict:
        """
        This method will drop the duplicate rows in the datasets.
        """
        for dataset_name, dataset in datasets.items():
            cls.log.log(
                message=f"\n\t[Started] - Duplicate rows removal for {dataset_name}.",
                enable_logging=cls.enable_logging,
            )
            unique = set()
            column_names = dataset.column_names
            cls.log.log(
                message=f"\t\tNo.of rows in {dataset_name} : {len(dataset)}.",
                enable_logging=cls.enable_logging,
            )
            unique_dataset = dataset.filter(
                lambda example: not tuple(example[col] for col in column_names)
                in unique
                and not unique.add(tuple(example[col] for col in column_names))
            )
            no_of_duplicate_rows = len(dataset) - len(unique_dataset)
            cls.log.log(
                message=f"\t\tRemoved {no_of_duplicate_rows} duplicate rows in {dataset_name}.",
                enable_logging=cls.enable_logging,
            )
            datasets[dataset_name] = unique_dataset
            cls.log.log(
                message=f"\t[Completed] - Duplicate rows removal for {dataset_name}.",
                enable_logging=cls.enable_logging,
            )
        return datasets

    @classmethod
    def convert_string_labels_to_integers(cls, datasets) -> dict:
        """
        This method converts the string lables in the sujet_finance dataset to integer labels.
        """
        sentiment_mapping = cls.config.get_sentiment_mapping()
        yes_no_mapping = cls.config.get_yes_no_mapping()

        sentiment_analysis_conversion = [
            "sujet_finance_sentiment_analysis",
            "sujet_finance_sentiment_analysis_fine_tuning",
        ]
        yes_no_question_conversion = [
            "sujet_finance_yes_no_question",
            "sujet_finance_yes_no_question_fine_tuning",
        ]

        for dataset_name in sentiment_analysis_conversion:
            cls.log.log(
                message=f"\t[Started] - Sentiment Analysis conversion for {dataset_name} dataset.",
                enable_logging=cls.enable_logging,
            )
            datasets[dataset_name] = datasets[dataset_name].map(
                cls.helpers.replace_string_with_int,
                fn_kwargs={"mapping": sentiment_mapping, "column_to_modify": "label"},
            )
            cls.log.log(
                message=f"\t\tMapping : {sentiment_mapping}",
                enable_logging=cls.enable_logging,
            )
            cls.log.log(
                message=f"\t[Completed] - Sentiment Analysis conversion for {dataset_name} dataset.",
                enable_logging=cls.enable_logging,
            )

        for dataset_name in yes_no_question_conversion:
            cls.log.log(
                message=f"\t[Started] - Yes/No conversion for {dataset_name} dataset.",
                enable_logging=cls.enable_logging,
            )
            datasets[dataset_name] = datasets[dataset_name].map(
                cls.helpers.replace_string_with_int,
                fn_kwargs={"mapping": yes_no_mapping, "column_to_modify": "label"},
            )
            cls.log.log(
                message=f"\t\tMapping : {yes_no_mapping}",
                enable_logging=cls.enable_logging,
            )
            cls.log.log(
                message=f"\t[Completed] - Yes/No conversion for {dataset_name} dataset.",
                enable_logging=cls.enable_logging,
            )
        return datasets

    @classmethod
    def seperate_datasets(cls, datasets) -> dict:
        """
        This function seperate creates datasets for finetuning and for normal flow.
        It also saves in local in order to avoid repeating the process.
        """
        temp_dict = {}
        for dataset_name, dataset in datasets.items():
            dataset_split = dataset.train_test_split(
                test_size=cls.fine_tune_split_size, seed=cls.seed
            )
            fine_tuned_dataset = dataset_split["test"].shuffle(seed=cls.seed)
            non_fine_tuned_dataset = dataset_split["train"].shuffle(seed=cls.seed)
            temp_dict[f"{dataset_name}_fine_tuning"] = fine_tuned_dataset
            temp_dict[f"{dataset_name}"] = non_fine_tuned_dataset

        datasets.update(temp_dict)

        return datasets

    @classmethod
    def split_text_and_label(cls, question):
        """
        This function performs the yes no preprocessing.
        """
        # Use regex to find the last occurrence of " yes" or " no" (case-insensitive) at the end
        match = re.search(r"\s(yes|no)$", question.strip(), re.IGNORECASE)
        if match:
            # Split the question into 'text' and 'label'
            text = question[: match.start()].strip()  # Everything before "yes" or "no"
            label = match.group(1).lower()
            return text, label
        return question, None  # If no match is found, return the full question and None

    @classmethod
    def explode_yes_no_question_data(cls, dataset):
        """
        This function explodes the rows in the yes no question dataset.
        """
        df = dataset.to_pandas()
        df["text_w_final_answer"] = df.apply(
            lambda row: row["text"].strip() + f" {row['label'].strip()}", axis=1
        )
        # Split the 'text_w_final_answer' by '\n' to create a list of questions for each row
        df["questions_split"] = df["text_w_final_answer"].str.split("\n\n")
        # Explode the 'questions_split' list into separate rows
        df_exploded = df.explode("questions_split")
        df_exploded = df_exploded.rename(columns={"questions_split": "question"})
        df_exploded[["text", "label"]] = df_exploded["question"].apply(
            lambda x: pd.Series(cls.split_text_and_label(x))
        )
        final_yes_no_df = df_exploded[["text", "label"]]
        final_yes_no_df = final_yes_no_df.dropna()
        final_yes_no_df = final_yes_no_df.sample(frac=1).head(
            35000
        )  # sample 35,000 rows

        return_dataset = Dataset.from_pandas(final_yes_no_df).remove_columns(
            "__index_level_0__"
        )
        return return_dataset

    @classmethod
    def preprocess(cls, save_data_in_local: bool, read_data_from_local: bool) -> dict:
        """
        This function in the starting point for the preprocessing of the datasets and return the datasets.
        """
        # Load the logger
        logger = cls.log

        if read_data_from_local:
            datasets = {}
            datasets_to_load = cls.config.get_local_datasets_names()
            for dataset_name in datasets_to_load:
                logger.log(
                    message=f"\n[Started] - Loading the {dataset_name} dataset from local.",
                    enable_logging=cls.enable_logging,
                )
                datasets[dataset_name] = cls.helpers.read_dataset_from_local(
                    dataset_name=dataset_name
                )
                logger.log(
                    message=f"[Completed] - Loading the {dataset_name} dataset from local.",
                    enable_logging=cls.enable_logging,
                )
            return datasets
        else:
            print("\n[Started] - Preprocessing the datasets.")
            # 1. Load the financial_phrasebank and Sujet-Finance-Instruct-177k from hugging face.
            logger.log(
                message="\n[Started] - Load the Sujet-Finance-Instruct-177k dataset from hugging face.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.load_datasets()
            logger.log(
                message="[Completed] - Load the Sujet-Finance-Instruct-177k dataset from hugging face.",
                enable_logging=cls.enable_logging,
            )

            # 2. Filter out the selected task types from the Sujet-Finance-Instruct-177k dataset.
            logger.log(
                message="\n[Started] - Select the task types in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets["sujet_finance"] = cls.select_task_types_columns(
                dataset=cls.temp_datasets["sujet_finance"]
            )
            logger.log(
                message="[Completed] - Select the task types in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )

            # 3. Convert the labels, texts into lowercase in Sujet-Finance-Instruct-177k dataset.
            logger.log(
                message="\n[Started] - Convert the column vaules to lower case in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.convert_lower_case(datasets=cls.temp_datasets)
            logger.log(
                message="[Completed] - Convert the column vaules to lower case in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )

            # 4. Rename column names in Sujet-Finance-Instruct-177k dataset.
            logger.log(
                message="\n[Started] - Rename the column names in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.rename_column_names(datasets=cls.temp_datasets)
            logger.log(
                message="[Completed] - Rename the column names in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )

            # 5. Seggregate the Sujet-Finance-Instruct-177k dataset according to the task types.
            logger.log(
                message="\n[Started] - Seggregate the data sets based on task types in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.seggregate_sujet_task_types(
                datasets=cls.temp_datasets
            )
            logger.log(
                message="[Completed] - Seggregate the data sets based on task types in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )

            # 6. Check for the null rows and drop them if required.
            logger.log(
                message="\n[Started] - Remove null rows from the datasets.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.drop_null_rows(datasets=cls.temp_datasets)
            logger.log(
                message="[Completed] - Remove null rows from the datasets.",
                enable_logging=cls.enable_logging,
            )

            # 7. Check for the duplicate rows and drop them if required.
            logger.log(
                message="\n[Started] - Remove duplicate rows from the datasets.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.drop_duplicate_rows(datasets=cls.temp_datasets)
            logger.log(
                message="[Completed] - Remove duplicate rows from the datasets.",
                enable_logging=cls.enable_logging,
            )

            # 8. Peform preprocessing of yes no question data, i.e, exploding the rows.
            logger.log(
                message="\n[Started] - Perform exploding the rows for yes no question dataset.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets["sujet_finance_yes_no_question"] = (
                cls.explode_yes_no_question_data(
                    dataset=cls.temp_datasets["sujet_finance_yes_no_question"]
                )
            )
            logger.log(
                message="[Completed] - Perform exploding the rows for yes no question dataset.",
                enable_logging=cls.enable_logging,
            )

            # 9. Create seperate datasets for fine tuning and normal flow.
            logger.log(
                message="\n[Started] - Create seperate datasets for fine tuning and normal flow.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.seperate_datasets(datasets=cls.temp_datasets)
            logger.log(
                message="[Completed] - Create seperate datasets for fine tuning and normal flow",
                enable_logging=cls.enable_logging,
            )

            # 10. Replace the string labels with integers in Sujet-Finance-Instruct-177k dataset.
            logger.log(
                message="\n[Started] - Convert the string labels to integers in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )
            cls.temp_datasets = cls.convert_string_labels_to_integers(
                datasets=cls.temp_datasets
            )
            logger.log(
                message="[Completed] - Convert the string labels to integers in the Sujet-Finance-Instruct-177k dataset.",
                enable_logging=cls.enable_logging,
            )

            # 11. Return the final required datasets.
            datasets = {
                "sujet_finance": cls.temp_datasets["sujet_finance"],
                "sentiment_analysis": cls.temp_datasets[
                    "sujet_finance_sentiment_analysis"
                ],
                "yes_no_question": cls.temp_datasets["sujet_finance_yes_no_question"],
                "sujet_finance_fine_tuning": cls.temp_datasets[
                    "sujet_finance_fine_tuning"
                ],
                "sentiment_analysis_fine_tuning": cls.temp_datasets[
                    "sujet_finance_sentiment_analysis_fine_tuning"
                ],
                "yes_no_question_fine_tuning": cls.temp_datasets[
                    "sujet_finance_yes_no_question_fine_tuning"
                ],
            }
            if save_data_in_local:
                for dataset_name, dataset in datasets.items():
                    logger.log(
                        message=f"\t[Started] - Saving the {dataset_name} dataset to local directory.",
                        enable_logging=cls.enable_logging,
                    )
                    cls.helpers.save_dataset(dataset=dataset, file_name=dataset_name)
                    logger.log(
                        message=f"\t[Completed] - Saving the {dataset_name} dataset to local directory.",
                        enable_logging=cls.enable_logging,
                    )
            print("\n[Completed] - Preprocessing the datasets.")
            return datasets
