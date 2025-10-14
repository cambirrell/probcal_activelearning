"""Logging utilities for active learning experiments.

This module provides logging functions to track dataset states, validation metrics,
and experiment progress throughout active learning iterations.
"""

import logging
from datetime import datetime
from pathlib import Path

from probcal.data_modules.probcal_datamodule import ProbcalDataModule


def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Set up logging for active learning experiment.

    Args:
        log_dir: Directory to save log file
        experiment_name: Name of the experiment

    Returns:
        Configured logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_al_{timestamp}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],  # Also print to console
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created at: {log_file}")
    return logger


def log_dataset_state(
    logger: logging.Logger,
    iteration: int,
    datamodule: ProbcalDataModule,
    stage: str,
    extra_info: dict = None,
):
    """Log the current state of datasets.

    Args:
        logger: Logger instance
        iteration: Current active learning iteration
        datamodule: The datamodule containing train/unlabeled splits
        stage: Description of current stage (e.g., "INITIAL", "AFTER_TRAINING")
        extra_info: Optional dict with additional info to log
    """
    train_size = len(datamodule.train) if hasattr(datamodule, "train") else 0
    unlabeled_size = len(datamodule.unlabeled) if hasattr(datamodule, "unlabeled") else 0
    val_size = len(datamodule.val) if hasattr(datamodule, "val") else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Iteration {iteration} - Stage: {stage}")
    logger.info(f"{'='*60}")
    logger.info(f"  Training set size:   {train_size:6d}")
    logger.info(f"  Unlabeled pool size: {unlabeled_size:6d}")
    logger.info(f"  Validation set size: {val_size:6d}")
    logger.info(f"  Total samples:       {train_size + unlabeled_size + val_size:6d}")

    if extra_info:
        logger.info(f"\n  Additional Info:")
        for key, value in extra_info.items():
            logger.info(f"    {key}: {value}")
    logger.info(f"{'='*60}\n")


def validate_dataset_state(
    datamodule: ProbcalDataModule, expected_total: int, logger: logging.Logger = None
):
    """Validate that dataset sizes make sense.

    Args:
        datamodule: The datamodule to validate
        expected_total: Expected total number of samples
        logger: Optional logger for warnings

    Raises:
        AssertionError: If dataset sizes don't sum to expected total
    """
    train_size = len(datamodule.train)
    unlabeled_size = len(datamodule.unlabeled)
    val_size = len(datamodule.val)

    total = train_size + unlabeled_size + val_size

    if total != expected_total:
        error_msg = (
            f"Dataset size mismatch! "
            f"Train({train_size}) + Unlabeled({unlabeled_size}) + Val({val_size}) "
            f"= {total}, expected {expected_total}"
        )
        if logger:
            logger.error(error_msg)
        raise AssertionError(error_msg)
