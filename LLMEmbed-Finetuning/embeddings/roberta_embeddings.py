"""
    This module contains the class to extract embeddings using Roberta LLM.
"""

# Local Imports
from config import Config
from logger import Logger
from helpers import Helpers
from models.model import LLM

# General Imports
import torch
from tqdm import trange


class Roberta:
    """
    This class contains the methods to extract the embeddings using Roberta.
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
        cls.model_name = "FacebookAI/roberta-large"
        cls.device = cls.config.get_device()
        cls.enable_logging = enable_logging
        cls.use_finetuned_model = use_finetuned_model
        cls.max_length = 128

    @classmethod
    def extract_roberta_embeddings(
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
        This method performs the embeddings extractions using roberta.
        """

        cls.log.log(
            message=f"\n[Started] - Performing embeddings extraction using Roberta for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        if cls.use_finetuned_model:
            path = f"finetuned/roberta_embeddings/{task}/dataset_tensors/"
        else:
            path = f"base/roberta_embeddings/{task}/dataset_tensors/"

        sentences_reps = []
        step = 16  # Reduced batch size to save memory

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
                reps_batch = batch_outputs.last_hidden_state[:, 0, :]
            sentences_reps.append(reps_batch.cpu())

            # Clear cache and delete variables to free memory
            torch.cuda.empty_cache()
            del sentences_batch, sentences_batch_encoding, batch_outputs, reps_batch

        sentences_reps = torch.cat(sentences_reps)
        labels = torch.stack([torch.tensor(label) for label in labels])

        cls.helpers.save_embeddings(
            sentences_embeds=sentences_reps, labels=labels, file_path=path, mode=mode
        )

        cls.log.log(
            message=f"[Completed] - Performing embeddings extraction using Roberta for {task} on {mode} data.",
            enable_logging=cls.enable_logging,
        )
        del sentences_reps, labels
