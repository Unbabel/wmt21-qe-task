# -*- coding: utf-8 -*-
import warnings

import torch
from pytorch_lightning.metrics import Metric
from scipy.stats import kendalltau, pearsonr


class Kendall(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")
        
    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        assert predictions.shape == scores.shape
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            return torch.tensor(kendalltau(self.predictions, self.scores)[0], dtype=torch.float32)

class Pearson(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx="sum")
        
    def update(self, predictions: torch.Tensor, scores: torch.Tensor):
        assert predictions.shape == scores.shape
        self.predictions += predictions.cpu().tolist() if predictions.is_cuda else predictions.tolist()
        self.scores += scores.cpu().tolist() if scores.is_cuda else predictions.tolist()

    def compute(self):
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            return torch.tensor(pearsonr(self.predictions, self.scores)[0], dtype=torch.float32)
