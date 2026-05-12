"""
    This module contains the class to finetune the Bert LLM on sentiment analysis and yes no question tasks.
"""

# General Imports
import time
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
    BertTokenizer,
)

# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger
from models.model import LLM


class BertFineTune:
    """
    This class contains the methods to perform Bert finetuning.
    """

    @classmethod
    def __init__(cls, enable_logging: bool):
        """
        This method initializes the variables and other instances.
        """
        cls.config = Config()
        cls.helpers = Helpers()
        cls.log = Logger()
        cls.llm = LLM(enable_logging=enable_logging)
        cls.enable_logging = enable_logging
        cls.finetuned_model_name = "bert-large-uncased-finetune-finance"
        cls.max_length = 128
        cls.device = cls.config.get_device()
        cls.epochs = 3
        cls.base_model_name = "google-bert/bert-large-uncased"

    @classmethod
    def tokenize_data(cls, example, tokenizer, column_to_modify: str, max_length: int):
        """
        This function tokenizes the data.
        """
        return tokenizer(
            example[column_to_modify],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    @classmethod
    def finetune_bert(cls, task: str):
        """
        This method performs the fine tuning for the Llama2 model.
        """
        cls.helpers.clear_huggingface_cache()
        torch.cuda.empty_cache()

        data_path = (
            f"/home/exouser/Desktop/msml-group8-code-base/data/{task}_fine_tuning"
        )
        dataset = load_from_disk(data_path)
        # tokenizer, model = cls.llm.get_bert(use_finetuned_model=False, task=None)
        model = BertForSequenceClassification.from_pretrained(
            cls.base_model_name, num_labels=cls.config.get_no_of_classes()[task]
        )
        tokenizer = BertTokenizer.from_pretrained(cls.base_model_name)

        # Tokenize the sentiment analysis dataset
        tokenized_dataset = dataset.map(
            cls.tokenize_data,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "column_to_modify": "text",
                "max_length": cls.max_length,
            },
        )
        # Convert tokenized data into PyTorch tensors
        input_ids = torch.tensor(tokenized_dataset["input_ids"])
        attention_masks = torch.tensor(tokenized_dataset["attention_mask"])
        labels = torch.tensor(tokenized_dataset["label"])

        # Create a TensorDataset
        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
        model = model.to(cls.device)
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

        total_steps = len(tensor_dataset) * 3  # 3 epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        tensor_dataset_loader = DataLoader(tensor_dataset, batch_size=16)

        start_time = time.time()
        for epoch in range(cls.epochs):
            print(f"\n[Started] - Epoch {epoch+1}")
            model.train()
            total_loss = 0

            for batch in tensor_dataset_loader:
                b_input_ids = batch[0].to(cls.device)
                b_input_mask = batch[1].to(cls.device)
                b_labels = batch[2].to(cls.device)

                model.zero_grad()
                outputs = model(
                    input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            print(f"Epoch {epoch+1} Loss: {total_loss/len(tensor_dataset)}")
            print(f"[Completed] - Epoch {epoch+1}")

        end_time = time.time()
        # Calculate the time difference
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        formatted_time = f"{minutes:02}:{seconds:02}"
        print(f"Time taken for finetuning of bert on {task}: {formatted_time}")

        new_model = f"{cls.finetuned_model_name}-{task}"
        cls.helpers.save_finetuned_model(
            model=model, tokenizer=tokenizer, model_name=new_model
        )
