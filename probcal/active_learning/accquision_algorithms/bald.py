"""BALD (Bayesian Active Learning by Disagreement) acquisition function.

This module implements the BALD acquisition strategy for active learning, which
selects samples that maximize the mutual information between predictions and
model parameters. BALD captures epistemic uncertainty (model uncertainty due to
lack of training data) rather than aleatoric uncertainty (inherent noise).

References:
    Houlsby et al. "Bayesian Active Learning for Classification and Preference Learning"
    https://arxiv.org/abs/1112.5745
"""

from typing import List, Tuple
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probcal.active_learning.accquision_algorithms.accquire_label import AcquisitionFunction
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.enums import DatasetType


class BALDAcquisition(AcquisitionFunction):
    """BALD acquisition function for probabilistic regression.
    
    BALD (Bayesian Active Learning by Disagreement) computes the mutual information
    between predictions and model parameters:
    
        BALD(x) = H[y|x,D] - E_θ[H[y|x,θ]]
    
    Where:
        - H[y|x,D] is the predictive entropy (total uncertainty)
        - E_θ[H[y|x,θ]] is the expected entropy (average model uncertainty)
        - The difference captures epistemic uncertainty (model disagreement)
    
    High BALD scores indicate samples where different model parameters disagree,
    suggesting that acquiring this sample would reduce model uncertainty.
    
    Attributes:
        dataset_type: Type of dataset (image, tabular, text).
        device: Device to run computations on.
        num_mc_samples: Number of Monte Carlo samples for uncertainty estimation.
    """
    
    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.IMAGE,
        device: torch.device | None = None,
        num_mc_samples: int = 10,
        **kwargs,
    ):
        """Initialize BALD acquisition function.
        
        Args:
            dataset_type: Type of dataset being used.
            device: Device for computations. If None, uses CUDA if available.
            num_mc_samples: Number of Monte Carlo samples for uncertainty estimation.
                More samples give better estimates but are slower.
            **kwargs: Additional parameters passed to parent class.
        """
        super().__init__(**kwargs)
        self.dataset_type = dataset_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_mc_samples = num_mc_samples
    
    @torch.no_grad()
    def score(
        self,
        model: ProbabilisticRegressionNN,
        unlabeled_loader: DataLoader,
        labeled_loader: DataLoader | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute BALD scores for each unlabeled sample.
        
        Args:
            model: Probabilistic regression model to evaluate.
            unlabeled_loader: DataLoader for unlabeled samples. Must return
                (inputs, targets, indices) tuples.
            labeled_loader: DataLoader for labeled samples (not used for BALD).
            **kwargs: Additional parameters (unused).
        
        Returns:
            Tuple of (scores, indices):
                - scores: (N,) tensor where N is total number of unlabeled samples.
                    Higher values indicate higher epistemic uncertainty.
                - indices: (N,) tensor of original dataset indices.
        """
        # Validate dataloader format
        self.validate_dataloader(unlabeled_loader)
        
        model.to(self.device)
        model.eval()
        
        all_scores = []
        all_indices = []
        
        # Process each batch
        for batch in tqdm(unlabeled_loader, desc="Computing BALD scores", leave=False):
            inputs, _, batch_indices = batch
            inputs = inputs.clone().detach().to(self.device)
            
            # Compute BALD score for each sample in the batch
            batch_scores = self._compute_bald_for_batch(model, inputs)
            
            all_scores.append(batch_scores)
            all_indices.append(batch_indices)
        
        # Concatenate all batches
        scores = torch.cat(all_scores)
        indices = torch.cat(all_indices)
        
        return scores, indices
    
    def _compute_bald_for_batch(
        self,
        model: ProbabilisticRegressionNN,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BALD scores for a batch of inputs.
        
        Args:
            model: The probabilistic regression model.
            inputs: (batch_size, *input_shape) tensor of inputs.
        
        Returns:
            (batch_size,) tensor of BALD scores.
        """
        batch_size = len(inputs)
        
        # Step 1: Sample T predictions from the model
        # For each input, we get T samples of (mean, variance)
        means = []  # Will be (T, batch_size)
        variances = []  # Will be (T, batch_size)
        
        for _ in range(self.num_mc_samples):
            y_hat = model.predict(inputs)  # Get distribution parameters
            
            # Sample from the predictive distribution
            # For Gaussian models: y_hat contains mean and log_variance
            if hasattr(y_hat, 'mean') and hasattr(y_hat, 'variance'):
                mean = y_hat.mean
                variance = y_hat.variance
            elif hasattr(y_hat, 'loc') and hasattr(y_hat, 'scale'):
                # For distributions with loc/scale
                mean = y_hat.loc
                variance = y_hat.scale ** 2
            else:
                # Fallback: assume y_hat is a tensor [mean, log_var] stacked
                # This depends on your model's output format
                # Adjust based on your ProbabilisticRegressionNN implementation
                if y_hat.shape[-1] == 2:
                    mean = y_hat[..., 0]
                    log_var = y_hat[..., 1]
                    variance = torch.exp(log_var)
                else:
                    # If only mean is returned, sample to estimate variance
                    mean = y_hat.squeeze()
                    samples = model.sample(y_hat, num_samples=10)
                    variance = samples.var(dim=0)
            
            means.append(mean.cpu())
            variances.append(variance.cpu())
        
        # Convert to tensors: (T, batch_size)
        means = torch.stack(means)
        variances = torch.stack(variances)
        
        # Step 2: Compute Predictive Entropy H[y|x,D]
        # This is the entropy of the mixture of Gaussians
        predictive_entropy = self._compute_predictive_entropy(means, variances)
        
        # Step 3: Compute Expected Entropy E_θ[H[y|x,θ]]
        # This is the average entropy of individual models
        expected_entropy = self._compute_expected_entropy(variances)
        
        # Step 4: BALD = Predictive Entropy - Expected Entropy
        bald_scores = predictive_entropy - expected_entropy
        
        return bald_scores
    
    def _compute_predictive_entropy(
        self,
        means: torch.Tensor,
        variances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predictive entropy H[y|x,D].
        
        For a mixture of Gaussians (from T model samples), we approximate the
        entropy using the variance of the mixture:
        
            Var[y|x,D] = E[Var[y|x,θ]] + Var[E[y|x,θ]]
                       = (1/T)Σ_t σ²_t + (1/T)Σ_t μ²_t - μ̄²
        
        Then: H[y|x,D] = 0.5 * log(2πe * Var[y|x,D])
        
        Args:
            means: (T, batch_size) tensor of predicted means.
            variances: (T, batch_size) tensor of predicted variances.
        
        Returns:
            (batch_size,) tensor of predictive entropies.
        """
        # Mean of means (predictive mean)
        mean_of_means = means.mean(dim=0)  # (batch_size,)
        
        # Variance of the mixture
        # Var = E[Var] + Var[E]
        mean_of_variances = variances.mean(dim=0)  # E[Var]
        variance_of_means = ((means - mean_of_means) ** 2).mean(dim=0)  # Var[E]
        
        predictive_variance = mean_of_variances + variance_of_means
        
        # Entropy of Gaussian: 0.5 * log(2πe * σ²)
        # Add small epsilon for numerical stability
        eps = 1e-8
        predictive_entropy = 0.5 * torch.log(2 * math.pi * math.e * (predictive_variance + eps))
        
        return predictive_entropy
    
    def _compute_expected_entropy(
        self,
        variances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expected entropy E_θ[H[y|x,θ]].
        
        This is the average entropy across individual model predictions:
        
            E_θ[H[y|x,θ]] = (1/T) Σ_t H[y|x,θ_t]
                          = (1/T) Σ_t 0.5 * log(2πe * σ²_t)
        
        Args:
            variances: (T, batch_size) tensor of predicted variances.
        
        Returns:
            (batch_size,) tensor of expected entropies.
        """
        # Entropy of each Gaussian: 0.5 * log(2πe * σ²)
        eps = 1e-8
        individual_entropies = 0.5 * torch.log(2 * math.pi * math.e * (variances + eps))
        
        # Average across models (T samples)
        expected_entropy = individual_entropies.mean(dim=0)  # (batch_size,)
        
        return expected_entropy
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BALDAcquisition("
            f"dataset_type={self.dataset_type.value}, "
            f"num_mc_samples={self.num_mc_samples})"
        )
