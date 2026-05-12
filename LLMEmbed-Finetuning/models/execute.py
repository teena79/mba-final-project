"""
    This module contains the class to execute the models.
"""

# Local Imports
from config import Config
from helpers import Helpers
from logger import Logger
from models import DownstreamModel
from dataset import Data

# General Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
from collections import defaultdict


class Execute:
    """
    This class contains the methods execute the models. It returns the valication metrics.
    """

    @classmethod
    def __init__(cls, enable_logging : bool, epochs : int, SIGMA : float, learning_rate : float):
        cls.log = Logger()
        cls.config = Config()
        cls.helpers = Helpers()
        cls.enable_logging = enable_logging
        cls.validation_metrics = defaultdict(dict)
        cls.cuda_no = 0
        cls.epochs = epochs
        cls.SIGMA = SIGMA
        cls.batch_size = 1024
        cls.lr = learning_rate
        cls.device = f"cuda:{cls.cuda_no}" if torch.cuda.is_available() else "cpu"
        cls.class_num_dict = cls.config.get_no_of_classes()
        cls.seed = 0

    @classmethod
    def train_multi_class(cls, dataloader, model, loss_fn, optimizer, task : str):
        """
        This method executes the models on train data for multi-class classification.
        """
        loss_list, acc_list, micro_f1_list, macro_f1_list = [], [], [], []
        for batch_i, batch_loader in enumerate(tqdm(dataloader)):
            batch_l, batch_b, batch_r, batch_y = batch_loader
            batch_l, batch_b, batch_r, batch_y = (
                batch_l.to(cls.device),
                batch_b.to(cls.device),
                batch_r.to(cls.device),
                batch_y.to(cls.device),
            )

            model.train()
            pred = model(batch_l.float(), batch_b.float(), batch_r.float())
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_y = torch.max(pred, 1).indices
            accuracy = accuracy_score(batch_y.cpu(), pred_y.cpu())
            micro_f1 = f1_score(batch_y.cpu(), pred_y.cpu(), average="micro")
            macro_f1 = f1_score(batch_y.cpu(), pred_y.cpu(), average="macro")
            loss = loss.cpu()

            loss_list.append(loss.item())
            acc_list.append(accuracy)
            micro_f1_list.append(micro_f1)
            macro_f1_list.append(macro_f1)

        cls.validation_metrics[f"{task}_train"]["loss"] = round(np.mean(loss_list), 4)
        cls.validation_metrics[f"{task}_train"]["accuracy"] = round(
            np.mean(acc_list), 4
        )
        cls.validation_metrics[f"{task}_train"]["micro_f1"] = round(
            np.mean(micro_f1_list), 4
        )
        cls.validation_metrics[f"{task}_train"]["macro_f1"] = round(
            np.mean(macro_f1_list), 4
        )

    @classmethod
    def test_multi_class(cls, dataloader, model, loss_fn, task : str):
        """
        This method executes the models on test data for multi-class classification.
        """
        avg_loss = 0
        total_pred, total_y = [], []

        for batch_i, batch_loader in enumerate(tqdm(dataloader)):
            batch_l, batch_b, batch_r, batch_y = batch_loader
            batch_l, batch_b, batch_r, batch_y = (
                batch_l.to(cls.device),
                batch_b.to(cls.device),
                batch_r.to(cls.device),
                batch_y.to(cls.device),
            )

            model.eval()
            with torch.no_grad():
                pred = model(batch_l.float(), batch_b.float(), batch_r.float())
                loss = loss_fn(pred, batch_y)
                loss = loss.to("cpu")
                avg_loss += loss.item()

            pred_y = torch.max(pred, 1).indices
            total_pred.append(pred_y.cpu())
            total_y.append(batch_y.cpu())

        avg_loss = avg_loss / (batch_i + 1)

        total_y = torch.cat(total_y)
        total_pred = torch.cat(total_pred)

        accuracy = accuracy_score(total_y, total_pred)
        micro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average="micro")
        macro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average="macro")

        cls.validation_metrics[f"{task}_test"]["avg_loss"] = round(avg_loss, 4)
        cls.validation_metrics[f"{task}_test"]["accuracy"] = round(accuracy, 4)
        cls.validation_metrics[f"{task}_test"]["micro_f1"] = round(micro_f1, 4)
        cls.validation_metrics[f"{task}_test"]["macro_f1"] = round(macro_f1, 4)

    @classmethod
    def train(cls, dataloader, model, loss_fn, optimizer, task : str):
        """
        This method executes the models on train data for binary classification.
        """
        loss_list, acc_list, f1_list = [], [], []

        for batch_i, batch_loader in enumerate(tqdm(dataloader)):
            batch_l, batch_b, batch_r, batch_y = batch_loader
            batch_l, batch_b, batch_r, batch_y = (
                batch_l.to(cls.device),
                batch_b.to(cls.device),
                batch_r.to(cls.device),
                batch_y.to(cls.device),
            )

            model.train()
            pred = model(batch_l.float(), batch_b.float(), batch_r.float())
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_y = torch.max(pred, 1).indices

            accuracy = accuracy_score(batch_y.cpu(), pred_y.cpu())
            f1 = f1_score(batch_y.cpu(), pred_y.cpu())
            loss = loss.cpu()

            loss_list.append(loss.item())
            acc_list.append(accuracy)
            f1_list.append(f1)

        cls.validation_metrics[f"{task}_train"]["loss"] = round(np.mean(loss_list), 4)
        cls.validation_metrics[f"{task}_train"]["accuracy"] = round(
            np.mean(acc_list), 4
        )
        cls.validation_metrics[f"{task}_train"]["f1_score"] = round(np.mean(f1_list), 4)

    @classmethod
    def test(cls, dataloader, model, loss_fn, task : str):
        """
        This method executes the models on test data for binary classification.
        """
        avg_loss = 0
        total_pred, total_y = [], []

        for batch_i, batch_loader in enumerate(tqdm(dataloader)):
            batch_l, batch_b, batch_r, batch_y = batch_loader
            batch_l, batch_b, batch_r, batch_y = (
                batch_l.to(cls.device),
                batch_b.to(cls.device),
                batch_r.to(cls.device),
                batch_y.to(cls.device),
            )

            model.eval()
            with torch.no_grad():
                pred = model(batch_l.float(), batch_b.float(), batch_r.float())
                loss = loss_fn(pred, batch_y)
                loss = loss.to("cpu")
                avg_loss += loss.item()

            pred_y = torch.max(pred, 1).indices
            total_pred.append(pred_y.cpu())
            total_y.append(batch_y.cpu())

        avg_loss = avg_loss / (batch_i + 1)

        total_y = torch.cat(total_y)
        total_pred = torch.cat(total_pred)
        accuracy = accuracy_score(total_y, total_pred)
        f1 = f1_score(total_y, total_pred)

        cls.validation_metrics[f"{task}_test"]["loss"] = round(avg_loss, 4)
        cls.validation_metrics[f"{task}_test"]["accuracy"] = round(accuracy, 4)
        cls.validation_metrics[f"{task}_test"]["f1_score"] = round(f1, 4)

    @classmethod
    def execute(cls, use_finetuned_embeddings: bool, task: str) -> dict:
        """
        The method executes the classification tasks and returns the validation metrics respectively.
        """
        torch.cuda.empty_cache()
        cls.helpers.set_seed(cls.seed)
        logger = cls.log
        logger.log(
            message="\n[Started] - Model Execution",
            enable_logging=cls.enable_logging,
        )
        class_num = cls.class_num_dict[task]
        model = DownstreamModel(class_num, cls.SIGMA).to(cls.device)
        loss_fn = nn.CrossEntropyLoss().to(cls.device)
        optimizer = optim.Adam(model.parameters(), cls.lr)
        logger.log(
            message=f"\n[Started] - Loading {task} train embeddings from local.",
            enable_logging=cls.enable_logging,
        )
        train_data = Data(
            task=task,
            mode="train",
            enable_logging=cls.enable_logging,
            use_finetuned_embeddings=use_finetuned_embeddings,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=cls.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        logger.log(
            message=f"[Completed] - Loading {task} train embeddings from local.",
            enable_logging=cls.enable_logging,
        )
        logger.log(
            message=f"\n[Started] - Loading {task} test embeddings from local.",
            enable_logging=cls.enable_logging,
        )
        test_data = Data(
            task=task,
            mode="test",
            enable_logging=cls.enable_logging,
            use_finetuned_embeddings=use_finetuned_embeddings,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=cls.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        logger.log(
            message=f"[Completed] - Loading {task} test embeddings from local.",
            enable_logging=cls.enable_logging,
        )
        if class_num > 2:
            logger.log(
                message=f"\n[Started] - {task} - Training.",
                enable_logging=cls.enable_logging,
            )
            for epoch in range(cls.epochs):
                model = model.to(cls.device)
                print(
                    f"--------------------------- Epoch {epoch+1}/{cls.epochs} ---------------------------"
                )
                cls.train_multi_class(
                    dataloader=train_loader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    task=task,
                )
            logger.log(
                message=f"[Completed] - {task} - Training.",
                enable_logging=cls.enable_logging,
            )
            logger.log(
                message=f"\n[Started] - {task} - Testing.",
                enable_logging=cls.enable_logging,
            )
            cls.test_multi_class(
                dataloader=test_loader, model=model, loss_fn=loss_fn, task=task
            )
            logger.log(
                message=f"[Completed] - {task} - Testing.",
                enable_logging=cls.enable_logging,
            )
        elif class_num == 2:
            logger.log(
                message=f"\n[Started] - {task} - Training.",
                enable_logging=cls.enable_logging,
            )
            for epoch in range(cls.epochs):
                model = model.to(cls.device)
                print(
                    f"--------------------------- Epoch {epoch+1}/{cls.epochs} ---------------------------"
                )
                cls.train(
                    dataloader=train_loader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    task=task,
                )
            logger.log(
                message=f"[Completed] - {task} - Training.",
                enable_logging=cls.enable_logging,
            )
            logger.log(
                message=f"\n[Started] - {task} - Testing.",
                enable_logging=cls.enable_logging,
            )
            cls.test(dataloader=test_loader, model=model, loss_fn=loss_fn, task=task)
            logger.log(
                message=f"[Completed] - {task} - Testing.",
                enable_logging=cls.enable_logging,
            )

        logger.log(
            message="[Completed] - Model Execution.",
            enable_logging=cls.enable_logging,
        )

        return cls.validation_metrics
