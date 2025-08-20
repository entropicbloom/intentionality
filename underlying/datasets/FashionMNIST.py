import os

import torch
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_path="./"):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        self.input_dim = 28**2

    def prepare_data(self):
        # download
        datasets.FashionMNIST(self.data_path, train=True, download=True)
        datasets.FashionMNIST(self.data_path, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        fashion_mnist_full = datasets.FashionMNIST(self.data_path, train=True, transform=self.transform)
        self.train, self.val = random_split(fashion_mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        self.test = datasets.FashionMNIST(self.data_path, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)