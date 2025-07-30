from pathlib import Path
from typing import Literal

import lightning as L
import torch
import numpy as np
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
    
    def unlabeled_partion_setup(self, partition_size: int):
        """
        Sets up the unlabeled dataset by partitioning the training data.
        """
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        if not (0 < partition_size < len(self.train)):
            raise ValueError("Partition size must be between 0 and 1.")
        # This should split the amount of training data into a partition for unlabeled data
        n = len(self.train)
        generator = torch.Generator().manual_seed(42)  # For reproducibility
        self.train, self.unlabeled = random_split(self.train, [partition_size, n - partition_size], generator=generator)
        self.al_setup = True
    
    def active_learning_add_label_data(self, data_to_label: list[tuple[torch.Tensor, torch.Tensor]]):
        print(f"Data label: {type(data_to_label)}")
        print(f"Data label: {type(data_to_label[0])}")
        print(f"Data label: {type(data_to_label[0][0])}")
        print(f"Data label: {data_to_label[0][0].shape}")
        print(f"Data label: {data_to_label[0][1].shape}")
        print(f"Data label: {data_to_label[0][1]}")
        print(f"Length of labeled: {len(data_to_label)}")
        if self.unlabeled is None:
            raise ValueError("The `unlabeled` attribute has not been set. Did you call `unlabeled_partion_setup` yet?")
        labeled_data = torch.utils.data.TensorDataset(
            torch.stack([item[0] for item in data_to_label]),
            torch.stack([item[1] for item in data_to_label])
        )
        print(f"data length: {len(labeled_data)}")
        print("Shape____")
        x, y = labeled_data[0]
        # print(x.shape, y.shape)
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        if not self.al_setup:
            raise ValueError("Active learning setup has not been done. Did you call `unlabeled_partion_setup` yet?")
        train_len_before = len(self.train)
        unlabeled_len_before = len(self.unlabeled)
        # print(f"Before: train={train_len_before}, unlabeled={unlabeled_len_before}")
        print(len(self.train), len(labeled_data))
        self.train = torch.utils.data.ConcatDataset([self.train, labeled_data])

        # Efficient removal using hashes
        def tensor_hash(x, y):
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)
            return (
                x.cpu().numpy().tobytes(),
                y.item() if y.numel() == 1 else tuple(y.cpu().numpy().tolist())
            )
        
        def as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.tensor(x)  # fallback for lists, scalars, etc.

        def tensors_equal(a, b, atol=1e-3):
            # print(f"a 0 {type(a[0])}")
            # print(f"b 0 {type(b[0])}")
            # print(f"a 1 {type(a[1])}")
            # print(f"b 1 {type(b[1])}")
            return torch.allclose(as_tensor(a[0]), as_tensor(b[0]), atol=atol) and torch.allclose(as_tensor(a[1]), as_tensor(b[1]), atol=atol)

        def to_tensor_dataset(dataset):
            xs = []
            ys = []
            for x, y in dataset:
                xs.append(torch.as_tensor(x))
                ys.append(torch.as_tensor(y))
            return torch.utils.data.TensorDataset(torch.stack(xs), torch.stack(ys))

        labeled_hashes = set(tensor_hash(as_tensor(x), as_tensor(y)) for x, y in data_to_label)
        print(f"Unique hashes in data_to_label: {len(labeled_hashes)}")
        keep_indices = []
        removed_count = 0
        print(f"Length of unlabeled {len(self.unlabeled)}")
        for i in range(len(self.unlabeled)):
            # print(f"label {i}")
            # self.unlabeled[i] = (self.unlabeled[i][0], torch.tensor(self.unlabeled[i][1]))
            # print(f"label {i} point 1")
            unlabeled_sample = self.unlabeled[i]
            found = False
            for labeled_sample in data_to_label:
                if tensors_equal(unlabeled_sample, labeled_sample):
                    # print(f"label {i} point 2")
                    found = True
                    break
            if not found:
                keep_indices.append(i)
            else:
                removed_count += 1
        print(f"Samples removed from unlabeled: {removed_count}")
        self.unlabeled = torch.utils.data.Subset(self.unlabeled, keep_indices)
        self.unlabeled = to_tensor_dataset(self.unlabeled)

        # print(f"After: train={len(self.train)}, unlabeled={len(self.unlabeled)}")
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
