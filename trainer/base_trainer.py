import torch
import logging
import argparse

from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseTrainer(object):
    def __init__(self,
                 epochs: int,
                 cfg: object,
                 dataset: torch.utils.data.Dataset,
                 model: torch.nn.Module,
                 loss_fn: callable,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 save_dir: str,
                 *args, **kwargs
                 ):
        self.cfg = cfg
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def before_train_epoch(self, *args, **kwargs):
        pass

    def after_train_epoch(self, *args, **kwargs):
        pass

    def before_train_iter(self, *args, **kwargs):
        pass

    def after_train_iter(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_iter(self):
        raise NotImplementedError
        
    @abstractmethod
    def validate_iter(self):
        raise NotImplementedError
        
    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.epochs):
            self.before_train_epoch(epoch=epoch)
            self.model.train()
            for batch_idx, data in tqdm(enumerate(self.dataset)):
                self.before_train_iter(epoch=epoch, batch_idx=batch_idx, data=data)
                self.train_iter(epoch=epoch, batch_idx=batch_idx, data=data)
                self.after_train_iter(epoch=epoch, batch_idx=batch_idx, data=data)
            self.after_train_epoch(epoch=epoch)