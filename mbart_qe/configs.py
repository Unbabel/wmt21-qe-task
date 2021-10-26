# -*- coding: utf-8 -*-
from argparse import Namespace
from typing import Union


class Config:
    def __init__(self, initial_data: dict) -> None:
        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])

    def namespace(self) -> Namespace:
        return Namespace(**self.to_dict())
    
    def to_dict(self) -> Namespace:
        return {
            name: getattr(self, name)
            for name in dir(self)
            if not callable(getattr(self, name)) and not name.startswith("__")
        }
    
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