"""CCE-based acquisition function for active learning.

This module implements the Conditional Calibration Error (CCE) as an uncertainty
metric for active learning. CCE measures how well a model's predictive distribution
matches the true conditional distribution, and can be used to identify samples
where the model is poorly calibrated.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probcal.active_learning.accquision_algorithms.accquire_label import AcquisitionFunction
from probcal.enums import DatasetType
from probcal.evaluation.calibration_evaluator import CalibrationEvaluator
from probcal.evaluation.calibration_evaluator import CalibrationEvaluatorSettings
from probcal.evaluation.calibration_evaluator import CCESettings
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN


class CCEAcquisition(AcquisitionFunction):
    """Acquisition function based on Conditional Calibration Error (CCE).

    This method computes a calibration-based uncertainty score for each unlabeled
    sample by measuring how well the model's predictive distribution aligns with
    a reference distribution from labeled data.

    Higher CCE scores indicate samples where the model is less well-calibrated,
    suggesting higher uncertainty.

    Attributes:
        dataset_type: Type of dataset (IMAGE, TABULAR, TEXT).
        cce_settings: Settings for CCE computation (kernels, lambda, num_mc_samples).
        device: Device to run computations on.
    """

    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.IMAGE,
        cce_settings: CCESettings | None = None,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Initialize CCE acquisition function.

        Args:
            dataset_type: Type of dataset being used.
            cce_settings: Optional CCE computation settings. If None, uses defaults.
            device: Device for computations. If None, uses CUDA if available.
            **kwargs: Additional parameters passed to parent class.
        """
        super().__init__(**kwargs)
        self.dataset_type = dataset_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up CCE settings
        if cce_settings is None:
            cce_settings = CCESettings(
                num_mc_samples=10,  # Use multiple MC samples for better uncertainty estimates
                input_kernel="polynomial",
                output_kernel="rbf",
                lmbda=0.1,
            )
        self.cce_settings = cce_settings

        # Initialize evaluator
        evaluator_settings = CalibrationEvaluatorSettings(
            dataset_type=self.dataset_type,
            device=self.device,
            cce_settings=self.cce_settings,
        )
        self.evaluator = CalibrationEvaluator(evaluator_settings)

    @torch.no_grad()
    def score(
        self,
        model: ProbabilisticRegressionNN,
        unlabeled_loader: DataLoader,
        labeled_loader: DataLoader | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CCE-based uncertainty scores for each unlabeled sample.

        Args:
            model: Probabilistic regression model to evaluate.
            unlabeled_loader: DataLoader for unlabeled samples. Must return
                (inputs, targets, indices) tuples.
            labeled_loader: DataLoader for labeled reference samples. Required
                for CCE computation.
            **kwargs: Additional parameters (unused for CCE).

        Returns:
            Tuple of (scores, indices):
                - scores: (N,) tensor where N is total number of unlabeled samples.
                    Higher values indicate higher uncertainty.
                - indices: (N,) tensor of original dataset indices.

        Raises:
            ValueError: If labeled_loader is None (CCE requires reference data).
        """
        if labeled_loader is None:
            raise ValueError("CCE acquisition requires a labeled reference set (labeled_loader).")

        # Validate dataloader formats
        self.validate_dataloader(unlabeled_loader)
        self.validate_dataloader(labeled_loader)

        model.to(self.device)
        model.eval()

        # Compute per-sample CCE scores
        scores, indices = self._compute_per_sample_cce(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
        )

        return scores, indices

    def _compute_per_sample_cce(
        self,
        model: ProbabilisticRegressionNN,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CCE score for each individual unlabeled sample.

        This is the core computation that scores each sample based on how well
        the model's predictions align with the reference distribution from
        labeled data.

        Args:
            model: The model to evaluate.
            labeled_loader: Reference labeled data.
            unlabeled_loader: Unlabeled samples to score.

        Returns:
            Tuple of (scores, indices) for all unlabeled samples.
        """
        # Get reference samples from labeled data using the public API
        x_ref, y_ref, x_prime_ref, y_prime_ref = (
            self.evaluator.get_reference_samples_for_active_learning(model, labeled_loader)
        )
        x_kernel, y_kernel = self.evaluator._get_kernel_functions(y_ref)

        all_scores = []
        all_indices = []

        # Process each batch in the unlabeled set
        for batch in tqdm(unlabeled_loader, desc="Computing CCE scores", leave=False):
            inputs, targets, batch_indices = batch
            # Clone the tensor to make it a normal tensor (removes inference-mode restrictions)
            inputs = inputs.clone().detach().to(self.device)

            # Encode inputs based on dataset type
            if self.dataset_type == DatasetType.TABULAR:
                encoded_inputs = self.evaluator._encode_tabular(inputs)
            elif self.dataset_type == DatasetType.IMAGE:
                encoded_inputs = self.evaluator._encode_image(inputs)
            else:
                encoded_inputs = self.evaluator._encode_text(inputs)

            # Get model predictions and samples
            y_hat = model.predict(inputs)

            # Generate MC samples for each input
            x_samples = torch.repeat_interleave(
                encoded_inputs,
                repeats=self.cce_settings.num_mc_samples,
                dim=0,
            )
            y_samples = model.sample(
                y_hat,
                num_samples=self.cce_settings.num_mc_samples,
            ).flatten()

            # Compute CCE for each sample in the batch individually
            # We treat each sample as its own "grid" point
            for i in range(len(inputs)):
                sample_encoded = encoded_inputs[i : i + 1]  # Keep batch dim

                # Get the MC samples for this specific input
                start_idx = i * self.cce_settings.num_mc_samples
                end_idx = (i + 1) * self.cce_settings.num_mc_samples
                sample_x_prime = x_samples[start_idx:end_idx]
                sample_y_prime = y_samples[start_idx:end_idx]

                # Import here to avoid circular dependency
                from probcal.evaluation.metrics import compute_mcmd_torch

                # Compute CCE for this single sample
                cce_score = compute_mcmd_torch(
                    grid=sample_encoded,
                    x=x_ref,
                    y=y_ref,
                    x_prime=sample_x_prime,
                    y_prime=sample_y_prime,
                    x_kernel=x_kernel,
                    y_kernel=y_kernel,
                    lmbda=self.cce_settings.lmbda,
                )

                all_scores.append(cce_score.item())
                all_indices.append(batch_indices[i].item())

        return torch.tensor(all_scores), torch.tensor(all_indices, dtype=torch.long)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CCEAcquisition("
            f"dataset_type={self.dataset_type.value}, "
            f"num_mc_samples={self.cce_settings.num_mc_samples}, "
            f"lmbda={self.cce_settings.lmbda})"
        )
