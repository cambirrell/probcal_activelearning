from pathlib import Path
from typing import Literal

import lightning as L
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split, Subset


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
        self.al_setup = False

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

    def _toggle_indices(self, dataset: Dataset, state: bool):
        """Helper to safely toggle index returning on the base dataset."""
        # A dataset can be a Subset, so we need the underlying dataset object
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        if hasattr(base_dataset, 'return_index'):
            base_dataset.return_index(state)


    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        self._toggle_indices(self.train, False)
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
        self._toggle_indices(self.val, False)
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
        self._toggle_indices(self.test, False)
        return self.get_dataloader(
            split="test",
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
    

    def unlabeled_partion_setup(self, partition_size: int):
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        if not (0 < partition_size < len(self.train)):
            raise ValueError("Partition size must be a value between 0 and the length of the training set.")

        n = len(self.train)
        all_indices = list(range(n))
        np.random.shuffle(all_indices)

        # Assign the index lists to the instance so they can be updated later
        self.train_indices = all_indices[:partition_size]
        self.unlabeled_indices = all_indices[partition_size:]


        # Keep a reference to the original, full dataset
        self.original_train_dataset = self.train
        self.train = torch.utils.data.Subset(self.original_train_dataset, self.train_indices)
        self.unlabeled = torch.utils.data.Subset(self.original_train_dataset, self.unlabeled_indices)
        self.al_setup = True
    
    def active_learning_add_label_data(self, indices_to_label: list[int]):
        # Use sets for efficient operations
        indices_to_label_set = set(indices_to_label)
        unlabeled_indices_set = set(self.unlabeled_indices)

        # Add to training set by extending the list
        self.train_indices.extend(list(indices_to_label_set))

        # Remove from unlabeled set using set difference
        self.unlabeled_indices = list(unlabeled_indices_set - indices_to_label_set)

        # Recreate the datasets with the updated indices
        self.train = Subset(self.original_train_dataset, self.train_indices)
        self.unlabeled = Subset(self.original_train_dataset, self.unlabeled_indices)

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
        self._toggle_indices(self.unlabeled, True)
        return DataLoader(
            self.unlabeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
