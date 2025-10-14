"""Active learning acquisition algorithms for sample selection.

This module provides various acquisition functions for active learning, including:
- CCE (Conditional Calibration Error): Calibration-based uncertainty
- BALD (Bayesian Active Learning by Disagreement): Coming soon
- BatchBALD: Coming soon
- Random: Random sampling baseline

Example usage:
    >>> from probcal.active_learning.accquision_algorithms import CCEAcquisition
    >>> from probcal.enums import DatasetType
    >>>
    >>> acquisition = CCEAcquisition(dataset_type=DatasetType.IMAGE)
    >>> selected_indices = acquisition.select_samples(
    ...     model=model,
    ...     unlabeled_loader=unlabeled_loader,
    ...     labeled_loader=labeled_loader,
    ...     num_samples=100
    ... )
"""

from probcal.active_learning.accquision_algorithms.accquire_label import AcquisitionFunction
from probcal.active_learning.accquision_algorithms.cce import CCEAcquisition
from probcal.active_learning.accquision_algorithms.uniform import UniformAcquisition
from probcal.active_learning.accquision_algorithms.bald import BALDAcquisition

__all__ = [
    "AcquisitionFunction",
    "CCEAcquisition",
    "UniformAcquisition",
    "BALDAcquisition",
]
