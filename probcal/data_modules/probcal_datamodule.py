from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split

from probcal.data_modules.bootstrap_mixin import BootstrapMixin


class ProbcalDataModule(L.LightningDataModule, BootstrapMixin):
    train: Dataset | None = None
    val: Dataset | None = None
    test: Dataset | None = None
    unlabeled: Dataset | None = None

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def setup(self, stage):
        raise NotImplementedError("Must be implemented by subclass.")

    def set_bootstrap_indices(self, split: Literal["val", "test"]):
        """Randomly generate indices that define a new bootstrap sample of the given split.
        
        Args:
            split (Literal["val", "test"]): The dataset split to sample from.
        
        Raises:
            AttributeError: If the specified split has not yet been set in this data module (happens in the `setup` method).
            ValueError: If an invalid split name is passed.
        """
        if split == "val":
            if self.val is None:
                raise AttributeError("The `val` attribute has not been set. Did you call `setup` yet?")
        elif split == "test":
            if self.test is None:
                raise AttributeError(
                    "The `test` attribute has not been set. Did you call `setup` yet?"
                )
        else:
            raise ValueError("Invalid split specified. Must be 'val' or 'test'.")

        n = len(self.val) if split == "val" else len(self.test)
        indices = torch.multinomial(torch.ones((n,)), num_samples=n, replacement=True)
        super().set_bootstrap_indices(split, indices)

    def clear_bootstrap_indices(self, split: Literal["val", "test"]):
        if split == "val":
            if self.val is None:
                raise ValueError("The `val` attribute has not been set. Did you call `setup` yet?")
        elif split == "test":
            if self.test is None:
                raise ValueError(
                    "The `test` attribute has not been set. Did you call `setup` yet?"
                )
        else:
            raise ValueError("Invalid split specified. Must be 'val' or 'test'.")
        super().set_bootstrap_indices(split, None)

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val is None:
            raise ValueError("The `val` attribute has not been set. Did you call `setup` yet?")
        return self.get_dataloader(
            split="val",
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test is None:
            raise ValueError("The `test` attribute has not been set. Did you call `setup` yet?")
        return self.get_dataloader(
            split="test",
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
    
    def unlabeled_partion_setup(self, partition_size: float):
        """
        Sets up the unlabeled dataset by partitioning the training data.
        """
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        if not (0 < partition_size < 1):
            raise ValueError("Partition size must be between 0 and 1.")
        # This should split the amount of training data into a partition for unlabeled data
        n = len(self.train)
        partition_size = int(n * partition_size)
        generator = torch.Generator().manual_seed(42)  # For reproducibility
        self.unlabeled, self.train = random_split(self.train, [partition_size, n - partition_size], generator=generator)
        self.al_setup = True
    
    def active_learning_add_label_data(self, data_to_label: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Adds labeled data to the training dataset for active learning, and 
        removes it from the unlabeled dataset.
        Args:
            data_to_label (list[tuple[torch.Tensor, torch.Tensor]]): A list of tuples where each tuple contains a data point and its corresponding label. Assume this data is unbatched"""
        if self.unlabeled is None:
            raise ValueError("The `unlabeled` attribute has not been set. Did you call `unlabeled_partion_setup` yet?")
        
        # Assuming data_to_label is a list of tensors
        labeled_data = torch.utils.data.TensorDataset(
            torch.stack([item[0] for item in data_to_label]),
            torch.stack([item[1] for item in data_to_label])
        )
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        if not self.al_setup:
            raise ValueError("Active learning setup has not been done. Did you call `unlabeled_partion_setup` yet?")
        self.train = torch.utils.data.ConcatDataset([self.train, labeled_data])
        
        # Remove the labeled data from the unlabeled dataset
        # TODO: Implement logic to remove the labeled data from the unlabeled dataset
        keep_indices = []
        for i in range(len(self.unlabeled)):
            if not any(torch.equal(self.unlabeled[i][0], item[0]) and self.unlabeled[i][1] == item[1] for item in data_to_label):
                keep_indices.append(i)
        self.unlabeled = torch.utils.data.Subset(self.unlabeled, keep_indices)
        return self.train_dataloader(), self.unlabeled_dataloader()



    def unlabeled_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the unlabeled dataset.
        This method should be implemented by subclasses to provide the unlabeled dataset.
        
        Returns:
            DataLoader: A DataLoader for the unlabeled dataset.
        """
        if self.unlabeled is None:
            raise ValueError("The `unlabeled` attribute has not been set. Did you call `unlabeled_partion_setup` yet?")
        return DataLoader(
            self.unlabeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
