# -*- coding: utf-8 -*-
import multiprocessing
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchnlp.utils import collate_tensors
from transformers import AdamW, MBartModel
from utils import Config

from mbart_qe.metrics import Pearson, Kendall
from mbart_qe.tokenizer import Tokenizer
from mbart_qe.scalar_mix import ScalarMixWithDropout
from mbart_qe.utils import move_to_cpu, move_to_cuda, lengths_to_mask
from mbart_qe.variance_loss import VarianceLoss
from mbart_qe.kl_loss import KLLoss

from tqdm import tqdm



class NMTEstimator(pl.LightningModule):

    class ModelConfig(Config):
        pretrained_model: str = "Unbabel/mbart50-large-m2m_mlqe-pe"

        monitor: str = "pearson"
        metric_mode: str = "max"
        loss: str = "varmse"

        # Optimizer
        learning_rate: float = 3.0e-5
        encoder_learning_rate: float = 1.0e-5
        keep_embeddings_frozen: bool = False
        keep_encoder_frozen: bool = False

        nr_frozen_epochs: Union[float, int] = 0.3
        dropout: float = 0.1
        hidden_size: int = 2048

        # Data configs
        train_path: str = None
        val_path: str = None
        test_path: str = None
        load_weights_from_checkpoint: Union[str, bool] = False
        
        # Training details
        batch_size: int = 4

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_path", "val_path", "test_path", "load_weights_from_checkpoint"]
        )
        self._hparams = self.hparams.hparams 
        self.model = MBartModel.from_pretrained(self._hparams.pretrained_model, output_hidden_states=True)
        self.tokenizer = Tokenizer(self._hparams.pretrained_model)

        output_dim = 1 if self._hparams.loss == "mse" else 2
        self.estimator = nn.Sequential(
            nn.Linear(self.model.config.hidden_size*2, self._hparams.hidden_size),
            nn.Tanh(),
            nn.Dropout(self._hparams.dropout),
            nn.Linear(self._hparams.hidden_size, output_dim),
        )

        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.model.config.decoder_layers+1,
            dropout=self._hparams.dropout,
            do_layer_norm=True,
        )

        if self._hparams.loss == "varmse":
            self.loss_fn = VarianceLoss()

        elif self._hparams.loss == "kl":
            self.loss_fn = KLLoss()

        else:
            self.loss_fn = nn.MSELoss()

        self.train_pearson = Pearson()
        self.dev_pearson = Pearson()
        self.train_kendall = Kendall()
        self.dev_kendall = Kendall()

        if self._hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if self._hparams.keep_embeddings_frozen:
            self.freeze_embeddings()
            
        self.mc_dropout = False

        if self._hparams.load_weights_from_checkpoint:
            self.load_weights(self._hparams.load_weights_from_checkpoint)

        self.epoch_nr = 0

    def load_weights(self, checkpoint: str) -> None:
        """Function that loads the weights from a given checkpoint file.
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.

        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        click.secho(f"Loading weights from {checkpoint}.", fg="red")
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)
    
    def configure_optimizers(self):
        self.epoch_total_steps = len(self.train_dataset) // (
            self._hparams.batch_size * max(1, self.trainer.num_gpus)
        )
        parameters = [
            {"params": self.estimator.parameters(), "lr": self._hparams.learning_rate},
            {"params": self.model.parameters(), "lr": self._hparams.encoder_learning_rate},
            {"params": self.scalar_mix.parameters(), "lr": self._hparams.learning_rate},
        ]
        optimizer = AdamW(
            parameters, lr=self._hparams.learning_rate, correct_bias=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self._hparams.metric_mode),
            "monitor": self._hparams.monitor
        }

    def freeze_embeddings(self) -> None:
        if self._hparams.keep_embeddings_frozen:
            print ("Keeping Embeddings Frozen!")
            for param in self.model.shared.parameters():
                param.requires_grad = False

        if self._hparams.keep_encoder_frozen:
            print ("Keeping Encoder Frozen!")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                

    def freeze_encoder(self) -> None:
        """ Freezes the pretrained NMT. """
        for param in self.model.parameters():
            param.requires_grad = False


    def unfreeze_encoder(self) -> None:
        """ Starts fine-tuning the pretrained NMT. """
        if self._frozen:
            if self.trainer.is_global_zero:
                click.secho("\nModel fine-tuning", fg="red")

            for param in self.model.parameters():
                param.requires_grad = True

            self._frozen = False
            if self._hparams.keep_embeddings_frozen:
                self.freeze_embeddings()

    def set_mc_dropout(self, value: bool):
        self.mc_dropout = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mt_input_ids: torch.Tensor,
        mt_eos_ids: torch.Tensor,
    ):

        def forward_pass(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            mt_input_ids: torch.Tensor,
            mt_eos_ids: torch.Tensor,
        ):
            model_output_mt = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                decoder_input_ids=mt_input_ids, 
            )
            #mt_hidden_states = model_output_mt.decoder_hidden_states[-3]
            mt_hidden_states = self.scalar_mix(
                model_output_mt.decoder_hidden_states, 
                lengths_to_mask(mt_eos_ids, device=mt_eos_ids.device)
            )

            eos_embeds, avg_embeds = [], []
            for i, token_index in  enumerate(mt_eos_ids):
                eos_embeds.append(mt_hidden_states[i, token_index-1, :])
                avg_embeds.append(mt_hidden_states[i, :token_index-1, :].mean(dim=0))
        
            eos_embeds = torch.stack(eos_embeds)
            avg_embeds = torch.stack(avg_embeds)
            mt_summary = torch.cat((eos_embeds, avg_embeds), dim=1)
            return self.estimator(mt_summary)
            
        if self.mc_dropout:
            self.train()
            mcd_outputs = torch.stack(
                [
                    forward_pass(
                        input_ids, 
                        attention_mask, 
                        mt_input_ids, 
                        mt_eos_ids,
                    )[:, 0] 
                    for _ in range(self.mc_dropout)
                ]
            )
            mcd_mean = mcd_outputs.mean(dim=0)
            return mcd_mean
        else:
            return forward_pass(input_ids, attention_mask, mt_input_ids, mt_eos_ids)
        

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_input, batch_target = batch
        predicted_scores = self.forward(**batch_input)
        
        if self._hparams.loss == "kl":
            loss_value = self.loss_fn(
                predicted_scores[:, 0].view(-1), 
                predicted_scores[:, 1].view(-1),
                batch_target["score"],
                batch_target["std"],
            )
        elif self._hparams.loss == "varmse":
            loss_value = self.loss_fn(
                predicted_scores[:, 0].view(-1), 
                predicted_scores[:, 1].view(-1),
                batch_target["score"]
            )
        else:
            loss_value = self.loss_fn(predicted_scores.view(-1), batch_target["score"])
            
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)
        
        if (
            self._hparams.nr_frozen_epochs < 1.0
            and self._hparams.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self._hparams.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value
    
    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
        *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if dataloader_idx == 0:
            batch_input, batch_target = batch
            predicted_scores = self.forward(**batch_input)
            
            if self._hparams.loss == "varmse" or self._hparams.loss == "kl":
                predicted_scores = predicted_scores[:, 0]

            if batch_target["score"].size()[0] > 1:
                self.log("train_kendall", self.train_kendall(predicted_scores.view(-1), batch_target["score"]))
                self.log("train_pearson", self.train_pearson(predicted_scores.view(-1), batch_target["score"]))
        
        if dataloader_idx == 1:
            batch_input, batch_target = batch
            predicted_scores = self.forward(**batch_input)

            if self._hparams.loss == "varmse" or self._hparams.loss == "kl":
                predicted_scores = predicted_scores[:, 0]
                
            if batch_target["score"].size()[0] > 1:
                self.log("kendall", self.dev_kendall(predicted_scores.view(-1), batch_target["score"]))
                self.log("pearson", self.dev_pearson(predicted_scores.view(-1), batch_target["score"]))
            
    def validation_epoch_end(self, *args, **kwargs) -> None:
        self.log("kendall", self.dev_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_kendall.reset()
        self.log("pearson", self.dev_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.dev_pearson.reset()

        self.log("train_pearson", self.train_pearson.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_pearson.reset()
        self.log("train_kendall", self.train_kendall.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.train_kendall.reset()  
        
    
    def on_train_epoch_end(self, *args, **kwargs) -> None:
        """Hook used to unfreeze encoder during training."""
        self.epoch_nr += 1
        if self.epoch_nr >= self._hparams.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False
        
        

# ------------------------------ DATA ------------------------------
    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        if self._hparams.loss == "kl":
            df = df[["src", "mt", "score", "std", "lp"]]
            df["std"] = df["std"].astype(float)
        else:
            df = df[["src", "mt", "score", "lp"]]
            df["TP"] = df["TP"].astype(float)

        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["lp"] = df["lp"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = collate_tensors(sample)
        inputs = self.tokenizer.batch_encode(
            sample["src"], sample["mt"],
            [lp.split("-")[0] for lp in sample["lp"]], 
            [lp.split("-")[1] for lp in sample["lp"]]
        )

        if inference:
            return inputs

        if self._hparams.loss == "kl":
            targets = {
                "score": torch.tensor(sample["score"], dtype=torch.float),
                "std": torch.tensor(sample["std"], dtype=torch.float),
            }
        else:
            targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
            
        return inputs, targets

    def setup(self, stage) -> None:
        self.train_dataset = self.read_csv(self._hparams.train_path)
        self.val_reg_dataset = self.read_csv(self._hparams.val_path)
        
        # Always validate the model with 2k examples from training to control overfit.
        train_subset = np.random.choice(a=len(self.train_dataset), size=2000)
        self.train_subset = Subset(self.train_dataset, train_subset)

        if self._hparams.test_path:
            self.test_dataset = self.read_csv(self._hparams.test_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self._hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self._hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=multiprocessing.cpu_count(),
            ),
            DataLoader(
                dataset=self.val_reg_dataset,
                batch_size=self._hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=multiprocessing.cpu_count(),
            )
        ]

    def predict(self, samples, show_progress=True,  cuda=True, batch_size=2):
        if self.training:
            self.eval()

        if cuda and torch.cuda.is_available():
            self.to("cuda")

        batch_size = self._hparams.batch_size if batch_size < 1 else batch_size
        with torch.no_grad():
            batches = [
                samples[i : i + batch_size] for i in range(0, len(samples), batch_size)
            ]
            model_inputs = []
            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Preparing batches...",
                    dynamic_ncols=True,
                    leave=None,
                )
            for batch in batches:
                batch = self.prepare_sample(batch, inference=True)
                model_inputs.append(batch)
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

            if show_progress:
                pbar = tqdm(
                    total=len(batches),
                    desc="Scoring hypothesis...",
                    dynamic_ncols=True,
                    leave=None,
                )
            scores = []
            for model_input in model_inputs:
                if cuda and torch.cuda.is_available():
                    model_input = move_to_cuda(model_input)
                    model_out = self.forward(**model_input)
                    model_out = move_to_cpu(model_out)
                else:
                    model_out = self.forward(**model_input)

                model_scores = model_out.numpy().tolist()
                for i in range(len(model_scores)):
                    scores.append(model_scores[i])
                
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        assert len(scores) == len(samples)
        for i in range(len(scores)):
            samples[i]["predicted_score"] = scores[i]
        return samples, scores
