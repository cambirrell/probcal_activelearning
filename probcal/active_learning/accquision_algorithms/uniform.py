"""Uniform (random) acquisition function for active learning.

This module implements a uniform random sampling strategy as a baseline for
active learning. Samples are selected uniformly at random from the unlabeled
pool, providing a simple baseline to compare against more sophisticated
uncertainty-based methods.
"""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from probcal.active_learning.accquision_algorithms.accquire_label import AcquisitionFunction
from probcal.enums import DatasetType
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN


class UniformAcquisition(AcquisitionFunction):
    """Random/uniform acquisition function for active learning baseline.

    This acquisition function randomly selects samples from the unlabeled pool
    with equal probability. It serves as a baseline to evaluate whether
    uncertainty-based methods provide any benefit over random sampling.

    This method is very fast since it requires no model inference or feature
    computation - it only needs to access sample indices.

    Attributes:
        dataset_type: Type of dataset (kept for interface consistency).
        device: Device for computations (kept for interface consistency).
        random_seed: Optional seed for reproducibility.
        rng: PyTorch random number generator.
    """

    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.IMAGE,
        device: torch.device | None = None,
        random_seed: int | None = None,
        **kwargs,
    ):
        """Initialize uniform acquisition function.

        Args:
            dataset_type: Type of dataset being used (not used, kept for consistency).
            device: Device for computations (not used, kept for consistency).
            random_seed: Seed for reproducibility. If None, uses current RNG state.
            **kwargs: Additional parameters passed to parent class.
        """
        super().__init__(**kwargs)
        self.dataset_type = dataset_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed

        # Set up random number generator for reproducibility
        if random_seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(random_seed)
        else:
            self.rng = None

    @torch.no_grad()
    def score(
        self,
        model: ProbabilisticRegressionNN,
        unlabeled_loader: DataLoader,
        labeled_loader: DataLoader | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign random scores to all unlabeled samples.

        Generates random scores from uniform distribution [0, 1) for each sample.
        Higher scores will be selected first by the select_samples method.

        Args:
            model: Model (not used for uniform sampling, kept for interface).
            unlabeled_loader: DataLoader for unlabeled samples. Must return
                (inputs, targets, indices) tuples.
            labeled_loader: DataLoader for labeled samples (not used).
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (scores, indices):
                - scores: (N,) tensor of random values from [0, 1) where N is
                    total number of unlabeled samples.
                - indices: (N,) tensor of original dataset indices.
        """
        # Validate dataloader format
        self.validate_dataloader(unlabeled_loader)

        all_scores = []
        all_indices = []

        # Collect all indices from unlabeled pool and generate random scores
        for batch in unlabeled_loader:
            _, _, batch_indices = batch

            # Generate random scores for this batch
            batch_size = len(batch_indices)
            if self.rng is not None:
                random_scores = torch.rand(batch_size, generator=self.rng)
            else:
                random_scores = torch.rand(batch_size)

            all_scores.append(random_scores)
            all_indices.append(batch_indices)

        # Concatenate all batches
        scores = torch.cat(all_scores)
        indices = torch.cat(all_indices)

        return scores, indices

    def select_samples(
        self,
        model: ProbabilisticRegressionNN,
        unlabeled_loader: DataLoader,
        labeled_loader: DataLoader | None,
        num_samples: int,
        **kwargs,
    ) -> List[int]:
        """Select samples uniformly at random.

        This is a more efficient implementation than calling score() + topk
        since it directly performs random selection without computing scores
        for all samples.

        Args:
            model: Model (not used for uniform sampling).
            unlabeled_loader: DataLoader for unlabeled samples.
            labeled_loader: DataLoader for labeled samples (not used).
            num_samples: Number of samples to select.
            **kwargs: Additional parameters (unused).

        Returns:
            List of selected sample indices, randomly chosen from unlabeled pool.
        """
        # Validate dataloader format
        self.validate_dataloader(unlabeled_loader)

        # Collect all indices efficiently
        all_indices = []
        for batch in unlabeled_loader:
            _, _, batch_indices = batch
            all_indices.append(batch_indices)

        all_indices = torch.cat(all_indices)

        # Ensure we don't try to select more samples than available
        num_to_select = min(num_samples, len(all_indices))

        # Random permutation and select first num_to_select
        if self.rng is not None:
            perm = torch.randperm(len(all_indices), generator=self.rng)
        else:
            perm = torch.randperm(len(all_indices))

        selected_positions = perm[:num_to_select]
        selected_indices = [all_indices[i].item() for i in selected_positions]

        return selected_indices

    def __repr__(self) -> str:
        """String representation."""
        seed_str = f"seed={self.random_seed}" if self.random_seed is not None else "no seed"
        return f"UniformAcquisition({seed_str})"
