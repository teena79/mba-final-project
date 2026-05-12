"""
This module contains the class that executes all the finetuning process.
"""

# Local Imports
from logger import Logger
from config import Config
from .llama2_finetuning import Llama2FineTune
from .bert_finetuning import BertFineTune
from .roberta_finetuning import RobertaFineTune


class FineTune:
    """
    This class contains the methods to finetune the LLM models.
    """

    @classmethod
    def __init__(cls, enable_logging):
        cls.log = Logger()
        cls.config = Config()
        cls.enable_logging = enable_logging
        cls.llama2_finetune = Llama2FineTune(enable_logging=enable_logging)
        cls.bert_finetune = BertFineTune(enable_logging=enable_logging)
        cls.roberta_finetune = RobertaFineTune(enable_logging=enable_logging)

    @classmethod
    def finetune(cls, llama2: bool, bert: bool, roberta: bool):
        """
        This method executes the fine tuning of LLM model on sentiment analysis and yes no question.
        """
        tasks = cls.config.get_selected_task_types()
        if llama2:
            for task in tasks:
                cls.log.log(
                    message=f"\n[Started] - Finetuning of Llama2 model on {task}.",
                    enable_logging=cls.enable_logging,
                )
                cls.llama2_finetune.finetune_llama2(task=task)

                cls.log.log(
                    message=f"[Completed] - Finetuning of Llama2 model on {task}",
                    enable_logging=cls.enable_logging,
                )
        if bert:
            for task in tasks:
                cls.log.log(
                    message=f"\n[Started] - Finetuning of Bert model on {task}.",
                    enable_logging=cls.enable_logging,
                )
                cls.bert_finetune.finetune_bert(task=task)

                cls.log.log(
                    message=f"[Completed] - Finetuning of Bert model on {task}",
                    enable_logging=cls.enable_logging,
                )
        if roberta:
            for task in tasks:
                cls.log.log(
                    message=f"\n[Started] - Finetuning of Roberta model on {task}.",
                    enable_logging=cls.enable_logging,
                )
                cls.roberta_finetune.finetune_roberta(task=task)

                cls.log.log(
                    message=f"[Completed] - Finetuning of Roberta model on {task}",
                    enable_logging=cls.enable_logging,
                )
