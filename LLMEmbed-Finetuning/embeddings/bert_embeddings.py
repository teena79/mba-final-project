"""
    This module contains the class to extract embeddings using Bert LLM.
"""

# Local Imports
from config import Config
from logger import Logger
from helpers import Helpers
from models.model import LLM

# General Imports
import torch
from tqdm import trange


class Bert:
    """
    This class contains the methods to extract the embeddings using Bert.
    """

    @classmethod
    def __init__(cls, enable_logging: bool, use_finetuned_model: bool):
        """
        This method initialized the variables that are used in this class.
        """
        cls.config = Config()
        cls.log = Logger()
        cls.helpers = Helpers()
        cls.llm = LLM(enable_logging=enable_logging)
        cls.model_name = "google-bert/bert-large-uncased"
        cls.device = cls.config.get_device()
        cls.enable_logging = enable_logging
        cls.use_finetuned_model = use_finetuned_model
        cls.max_length = 512

    @classmethod
    def extract_bert_embeddings(
        cls,
        mode: str,
        device: str,
        sentences: list,
        labels: list,
        task: str,
        model,
        tokenizer,
    ):
        """
        This method performs the embeddings extractions using bert.
        """

        cls.log.log(
            message=f"\n[Started] - Performing embeddings extraction using Bert for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        if cls.use_finetuned_model:
            path = f"finetuned/bert_embeddings/{task}/dataset_tensors/"
        else:
            path = f"base/bert_embeddings/{task}/dataset_tensors/"

        sentences_reps = []
        step = 64  # Reduced batch size to save memory
        for idx in trange(0, len(sentences), step):
            idx_end = idx + step
            if idx_end > len(sentences):
                idx_end = len(sentences)
            sentences_batch = sentences[idx:idx_end]

            sentences_batch_encoding = tokenizer(
                sentences_batch,
                return_tensors="pt",
                max_length=cls.max_length,
                padding="max_length",
                truncation=True,
            )
            sentences_batch_encoding = sentences_batch_encoding.to(device)

            with torch.no_grad():
                batch_outputs = model(**sentences_batch_encoding)
                reps_batch = batch_outputs.pooler_output
            sentences_reps.append(reps_batch.cpu())

            # Clear cache after processing each batch to free up memory
            del sentences_batch_encoding, batch_outputs, reps_batch
            torch.cuda.empty_cache()

        sentences_reps = torch.cat(sentences_reps)

        for idx in range(len(labels)):
            labels[idx] = torch.tensor(labels[idx])
        labels = torch.stack(labels)

        cls.helpers.save_embeddings(
            sentences_embeds=sentences_reps, labels=labels, file_path=path, mode=mode
        )

        cls.log.log(
            message=f"[Completed] - Performing embeddings extraction using Bert for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        del sentences_reps, labels
