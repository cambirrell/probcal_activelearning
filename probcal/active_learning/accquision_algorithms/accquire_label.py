"""Base class for active learning acquisition functions.

This module defines the interface that all acquisition functions must implement.
Each acquisition function scores unlabeled samples based on uncertainty or
informativeness to guide sample selection in active learning.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions in active learning.

    An acquisition function takes a model and unlabeled data, computes an
    uncertainty/informativeness score for each sample, and returns those scores
    along with the corresponding sample indices.
    """

    def __init__(self, **kwargs):
        """Initialize the acquisition function with optional hyperparameters.

        Args:
            **kwargs: Acquisition-specific hyperparameters (e.g., num_mc_samples).
        """
        self.kwargs = kwargs

    @abstractmethod
    def score(
        self,
        model: ProbabilisticRegressionNN,
        unlabeled_loader: DataLoader,
        labeled_loader: DataLoader | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute uncertainty scores for unlabeled samples.

        Args:
            model: The probabilistic regression model to use for scoring.
            unlabeled_loader: DataLoader for unlabeled samples. Must return
                (inputs, targets, indices) tuples.
            labeled_loader: Optional DataLoader for labeled samples (used by
                some methods like CCE that need reference distributions).
            **kwargs: Additional method-specific parameters.

        Returns:
            Tuple containing:
                - scores: (N,) tensor of uncertainty scores, where N is the total
                    number of unlabeled samples. Higher scores = more uncertain.
                - indices: (N,) tensor of original dataset indices corresponding
                    to each score.

        Raises:
            ValueError: If the dataloader format is incorrect.
        """
        raise NotImplementedError

    def select_samples(
        self,
        model: ProbabilisticRegressionNN,
        unlabeled_loader: DataLoader,
        labeled_loader: DataLoader | None,
        num_samples: int,
        **kwargs,
    ) -> List[int]:
        """Select the top-k most uncertain samples from the unlabeled pool.

        This is a convenience method that calls score() and returns the indices
        of the top-k samples with highest uncertainty.

        Args:
            model: The probabilistic regression model.
            unlabeled_loader: DataLoader for unlabeled samples.
            labeled_loader: Optional DataLoader for labeled samples.
            num_samples: Number of samples to select.
            **kwargs: Additional method-specific parameters.

        Returns:
            List of dataset indices for the selected samples.
        """
        # Get uncertainty scores for all unlabeled samples
        scores, indices = self.score(
            model=model, unlabeled_loader=unlabeled_loader, labeled_loader=labeled_loader, **kwargs
        )

        # Select top-k samples with highest uncertainty
        num_to_select = min(num_samples, len(scores))
        topk_positions = torch.topk(scores, k=num_to_select).indices

        # Map back to original dataset indices
        selected_indices = [indices[i].item() for i in topk_positions]

        return selected_indices

    def validate_dataloader(self, loader: DataLoader) -> None:
        """Validate that a dataloader returns the expected format.

        Args:
            loader: DataLoader to validate.

        Raises:
            ValueError: If the dataloader doesn't return (x, y, index) tuples.
        """
        try:
            sample_batch = next(iter(loader))
            if len(sample_batch) != 3:
                raise ValueError(
                    f"Dataloader must return (inputs, targets, indices) tuples, "
                    f"but got {len(sample_batch)} elements."
                )
            if sample_batch[2].dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                raise ValueError(
                    f"Indices must be integer tensors, but got dtype {sample_batch[2].dtype}."
                )
        except StopIteration:
            raise ValueError("Dataloader is empty.")

    def __repr__(self) -> str:
        """String representation of the acquisition function."""
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"
