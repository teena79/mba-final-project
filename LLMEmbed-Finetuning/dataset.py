"""
    This module contains the class to read the data (embeddings).
"""

# General Imports
import torch
from torch.utils.data import Dataset

# Local Imports
from config import Config
from logger import Logger


class Data(Dataset):
    """
    This class loads the extracted embeddings.
    """

    @classmethod
    def __init__(
        cls, task: str, mode: str, enable_logging: bool, use_finetuned_embeddings: bool
    ):
        """
        This method inializes the instances and other variables and load embeddings.
        """
        cls.config = Config()
        cls.log = Logger()
        cls.enable_logging = enable_logging
        cls.data = {}
        (
            l_sentences_path,
            b_sentences_path,
            r_sentences_path,
            labels_path,
        ) = cls.config.get_embeddings_path(
            task=task, mode=mode, use_finetuned_embeddings=use_finetuned_embeddings
        )

        cls.l_sentences_reps = torch.load(l_sentences_path)
        cls.b_sentences_reps = torch.load(b_sentences_path)
        cls.r_sentences_reps = torch.load(r_sentences_path)
        cls.labels = torch.load(labels_path)

        cls.sample_num = cls.labels.shape[0]

    def __getitem__(cls, index):
        return (
            cls.l_sentences_reps[index],
            cls.b_sentences_reps[index],
            cls.r_sentences_reps[index],
            cls.labels[index],
        )

    def __len__(cls):
        return cls.sample_num
