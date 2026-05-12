"""
    This module is the starting point for th embeddings extraction.
"""

# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger
from .bert_embeddings import Bert
from .roberta_embeddings import Roberta
from .llama2_embeddings import Llama2
from models.model import LLM

# General Imports
from sklearn.model_selection import train_test_split
import torch


class Embeddings:
    """
    This class perform the following tasks.
        1. Extract embeddings for the Sujet-Finance-Instruct-177k using Bert Model and save the embeddings as .pt file.
        2. Extract embeddings for the Sujet-Finance-Instruct-177k using Llama Model and save the embeddings as .pt file.
        3. Extract embeddings for the Sujet-Finance-Instruct-177k using Roberta Model and save the embeddings as .pt file.
    """

    @classmethod
    def __init__(cls, enable_logging: bool, use_finetuned_model: bool):
        """
        This method initializes objects and the dictionary to save the datasets.
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.log = Logger()
        cls.llm = LLM(enable_logging=enable_logging)
        cls.bert = Bert(enable_logging, use_finetuned_model)
        cls.roberta = Roberta(enable_logging, use_finetuned_model)
        cls.llama2 = Llama2(enable_logging, use_finetuned_model)
        cls.use_finetuned_model = use_finetuned_model
        cls.enable_logging = enable_logging
        cls.device = cls.config.get_device()
        cls.seed = 0

    @classmethod
    def extract(cls, datasets, bert: bool, roberta: bool, llama2: bool) -> dict:
        """
        This method extracts the embeddings using the LLM's.
        Extracting embeddings for only sentiment_analysis and yes_no_question from the dataset.
        Extraction will not be performed on the fine tuning datasets.
        """
        cls.log.log(
            message=f"\n[Started] - Embeddings extraction.",
            enable_logging=cls.enable_logging,
        )
        cls.helpers.set_seed(cls.seed)
        for dataset_name, dataset in datasets.items():
            sentences = dataset["text"]
            labels = dataset["label"]
            # train-test split
            (
                sentences_train,
                sentences_test,
                labels_train,
                labels_test,
            ) = train_test_split(sentences, labels, test_size=0.2, random_state=42)

            if bert:
                # Bert Training data emebeddings extraction
                cls.helpers.clear_huggingface_cache()
                torch.cuda.empty_cache()
                bert_tokenizer, bert_model = cls.llm.get_bert(
                    use_finetuned_model=cls.use_finetuned_model, task=dataset_name
                )
                bert_model.eval()
                cls.bert.extract_bert_embeddings(
                    mode="train",
                    device=cls.device,
                    sentences=sentences_train,
                    labels=labels_train,
                    task=dataset_name,
                    model=bert_model,
                    tokenizer=bert_tokenizer,
                )

                # Bert Testing data emebeddings extraction
                cls.bert.extract_bert_embeddings(
                    mode="test",
                    device=cls.device,
                    sentences=sentences_test,
                    labels=labels_test,
                    task=dataset_name,
                    model=bert_model,
                    tokenizer=bert_tokenizer,
                )

            if roberta:
                # Roberta Training emebeddings extraction
                cls.helpers.clear_huggingface_cache()
                torch.cuda.empty_cache()
                roberta_tokenizer, roberta_model = cls.llm.get_roberta(
                    use_finetuned_model=cls.use_finetuned_model, task=dataset_name
                )
                roberta_model.eval()
                cls.roberta.extract_roberta_embeddings(
                    mode="train",
                    device=cls.device,
                    sentences=sentences_train,
                    labels=labels_train,
                    task=dataset_name,
                    model=roberta_model,
                    tokenizer=roberta_tokenizer,
                )

                # Roberta Testing emebeddings extraction
                cls.roberta.extract_roberta_embeddings(
                    mode="test",
                    device=cls.device,
                    sentences=sentences_test,
                    labels=labels_test,
                    task=dataset_name,
                    model=roberta_model,
                    tokenizer=roberta_tokenizer,
                )
            if llama2:
                # Llama2 Training emebeddings extraction
                cls.helpers.clear_huggingface_cache()
                torch.cuda.empty_cache()
                llama2_tokenizer, llama2_model = cls.llm.get_llama2(
                    use_finetuned_model=cls.use_finetuned_model, task=dataset_name
                )
                llama2_model.eval()
                cls.llama2.extract_llama2_embeddings(
                    mode="train",
                    device=cls.device,
                    sentences=sentences_train,
                    labels=labels_train,
                    task=dataset_name,
                    model=llama2_model,
                    tokenizer=llama2_tokenizer,
                )

                # Llama2 Testing emebeddings extraction
                cls.llama2.extract_llama2_embeddings(
                    mode="test",
                    device=cls.device,
                    sentences=sentences_test,
                    labels=labels_test,
                    task=dataset_name,
                    model=llama2_model,
                    tokenizer=llama2_tokenizer,
                )

        cls.log.log(
            message=f"[Completed] - Embeddings extraction.",
            enable_logging=cls.enable_logging,
        )
