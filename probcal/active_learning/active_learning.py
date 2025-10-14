import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import math
import matplotlib.pyplot as plt
import torch
import yaml

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from probcal.enums import DatasetType
from probcal.data_modules.probcal_datamodule import ProbcalDataModule
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.utils.configs import ActiveLearningConfig
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from torch.utils.data import DataLoader

# Import logging utilities
from probcal.active_learning.active_learning_logger import (
    setup_logging,
    log_dataset_state,
    validate_dataset_state,
)


def train_samples(
    model: ProbabilisticRegressionNN,
    config: ActiveLearningConfig,
    datamodule: ProbcalDataModule,
    iteration: int,
    logger: logging.Logger,
):
    """Train the model on the sampled data.

    Args:
        model: Model to train
        config: Active learning configuration
        datamodule: Data module with train/val splits
        iteration: Current iteration number
        logger: Logger instance

    Returns:
        Tuple of (trained_model, validation_metrics)
    """
    fix_random_seed(config.random_seed)
    csv_logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)
    chkp_dir = config.chkp_dir / config.experiment_name / f"version_{iteration}"
    chkp_callbacks = get_chkp_callbacks(chkp_dir, config.chkp_freq)

    trainer = L.Trainer(
        accelerator=config.accelerator_type.value,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=math.ceil(config.num_epochs / 200),
        enable_model_summary=False,
        callbacks=chkp_callbacks,
        logger=csv_logger,
        precision=config.precision,
    )

    logger.info(f"Training model for {config.num_epochs} epochs...")
    trainer.fit(model=model, datamodule=datamodule)
    val_metrics = trainer.validate(model=model, datamodule=datamodule)

    return model, val_metrics


def get_acquisition_function(
    metric: str,
    dataset_type: DatasetType,
    device: torch.device,
    random_seed: int | None = None,
    num_mc_samples: int = 10,
):
    """Factory function to create acquisition functions.

    Args:
        metric: Name of the acquisition metric (e.g., 'cce', 'bald', 'uniform').
        dataset_type: Type of dataset being used.
        device: Device for computations.
        random_seed: Random seed for reproducibility (used by uniform/random methods).
        num_mc_samples: Number of MC samples for uncertainty estimation (BALD only).

    Returns:
        An instance of AcquisitionFunction.

    Raises:
        ValueError: If metric is not recognized.
    """
    from probcal.active_learning.accquision_algorithms import (
        CCEAcquisition,
        UniformAcquisition,
        BALDAcquisition,
    )

    if metric.lower() == "cce":
        return CCEAcquisition(dataset_type=dataset_type, device=device)
    elif metric.lower() in ("uniform", "random"):
        return UniformAcquisition(
            dataset_type=dataset_type, device=device, random_seed=random_seed
        )
    elif metric.lower() == "bald":
        return BALDAcquisition(
            dataset_type=dataset_type,
            device=device,
            num_mc_samples=num_mc_samples,
        )
    else:
        raise ValueError(
            f"Unknown acquisition metric: {metric}. "
            f"Supported metrics: ['cce', 'bald', 'uniform', 'random']"
        )


def select_samples(
    unlabeled_data: DataLoader,
    training_data: DataLoader,
    model: ProbabilisticRegressionNN,
    num_samples: int,
    metric: str,
    dataset_type: DatasetType,
    device: torch.device,
    logger: logging.Logger,
) -> List[int]:
    """Select samples from the unlabeled pool based on an uncertainty metric.

    Args:
        unlabeled_data: DataLoader for unlabeled samples.
        training_data: DataLoader for labeled reference samples.
        model: Probabilistic regression model.
        num_samples: Number of samples to select.
        metric: Name of acquisition metric to use.
        dataset_type: Type of dataset.
        device: Device for computations.
        logger: Logger instance.

    Returns:
        List of original dataset indices for selected samples.
    """
    logger.info(f"Computing uncertainty scores using {metric}...")

    # Get the appropriate acquisition function
    acquisition_fn = get_acquisition_function(
        metric=metric,
        dataset_type=dataset_type,
        device=device,
        random_seed=None,  # Could be passed from config if needed
        num_mc_samples=10,  # Could be passed from config if needed
    )

    # Select samples using the acquisition function
    selected_indices = acquisition_fn.select_samples(
        model=model,
        unlabeled_loader=unlabeled_data,
        labeled_loader=training_data,
        num_samples=num_samples,
    )

    logger.info(f"Selected {len(selected_indices)} samples")
    logger.info(f"First 10 selected indices: {selected_indices[:10]}")

    return selected_indices


def plot_results(results: List[dict], log_dir: Path, experiment_name: str):
    """Plot the results of the model as more data is sampled.

    Args:
        results: List of result dictionaries from each iteration
        log_dir: Directory to save plots
        experiment_name: Name of experiment for plot title
    """
    plt.figure(figsize=(10, 5))

    samples_labeled = [r["Samples_Labeled"] for r in results]
    val_losses = [
        r["Eval"][0].get("val_loss", float("nan")) if r["Eval"] else float("nan") for r in results
    ]

    plt.plot(samples_labeled, val_losses, marker="o")
    plt.title(f"Model Performance Over Time - {experiment_name}")
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Validation Loss")
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = log_dir / f"{experiment_name}_results_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def log_results(log_dir: Path, experiment_name: str, eval_results: List[dict]) -> None:
    """Produces a YAML file with a heading for each training round.

    Args:
        log_dir: Directory to save results
        experiment_name: Name of the experiment
        eval_results: List of result dictionaries
    """
    log_dict = {}
    for i, result in enumerate(eval_results):
        round_key = f"round_{i+1}"
        log_dict[round_key] = {
            "Batches_Labeled": result.get("Batches_Labeled"),
            "Samples_Labeled": result.get("Samples_Labeled"),
            "Eval": result.get("Eval"),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{experiment_name}_al_results_{timestamp}.yaml"
    with open(log_path, "w") as f:
        yaml.dump(log_dict, f)

    print(f"Active learning results saved to {log_path}")


def main(config: ActiveLearningConfig) -> None:
    """Main function to run the active learning experiment.

    Args:
        config: Active learning configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up logging
    logger = setup_logging(config.log_dir, config.experiment_name)
    logger.info(f"\n{'#'*60}")
    logger.info(f"# Starting Active Learning Experiment: {config.experiment_name}")
    logger.info(f"{'#'*60}\n")
    logger.info(f"Configuration:")
    logger.info(f"  Budget: {config.budget} samples")
    logger.info(f"  Samples per iteration: {config.samples_per_iteration}")
    logger.info(f"  Acquisition metric: {config.uncertainty_metric}")
    logger.info(f"  Dataset type: {config.dataset_type.value}")
    logger.info(f"  Device: {device}")

    # Initialize model and data
    model = get_model(config)
    datamodule = get_datamodule(
        config.dataset_type, config.dataset_path_or_spec, config.batch_size, config.num_workers
    )
    datamodule.setup("fit")
    datamodule.unlabeled_partion_setup(config.initial_labeled_partition)
    model.to(device)

    # Log initial state
    log_dataset_state(
        logger,
        iteration=0,
        datamodule=datamodule,
        stage="INITIAL",
        extra_info={
            "Initial partition size": config.initial_labeled_partition,
            "Budget": config.budget,
        },
    )

    # Store total for validation
    total_samples = len(datamodule.train) + len(datamodule.unlabeled) + len(datamodule.val)

    # Active learning loop
    eval_results = []
    iteration = 1
    cumulative_labeled = len(datamodule.train)
    # If Budget is None, set to total samples to avoid infinite loop
    if config.budget is None:
        config.budget = len(datamodule.train) + len(datamodule.unlabeled)
    while cumulative_labeled < config.budget and len(datamodule.unlabeled) > 0:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# ACTIVE LEARNING ITERATION {iteration}")
        logger.info(f"{'#'*60}\n")

        # Step 1: Train model
        logger.info(f"Step 1: Training model...")
        model, val_metrics = train_samples(model, config, datamodule, iteration, logger)
        val_loss = val_metrics[0].get("val_loss", "N/A") if val_metrics else "N/A"

        eval_results.append(
            {
                "Iteration": iteration,
                "Batches_Labeled": cumulative_labeled // config.batch_size,
                "Samples_Labeled": cumulative_labeled,
                "Eval": val_metrics,
            }
        )

        log_dataset_state(
            logger,
            iteration=iteration,
            datamodule=datamodule,
            stage="AFTER_TRAINING",
            extra_info={
                "Validation loss": (
                    f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
                ),
                "Cumulative labeled": cumulative_labeled,
                "Progress": f"{cumulative_labeled}/{config.budget} ({100*cumulative_labeled/config.budget:.1f}%)",
            },
        )

        # Check if unlabeled pool is exhausted
        training_data = datamodule.train_dataloader()
        unlabeled_data = datamodule.unlabeled_dataloader()

        if len(unlabeled_data) == 0:
            logger.info("Unlabeled pool exhausted. Stopping.")
            break

        # Step 2: Select samples
        num_to_select = min(
            config.samples_per_iteration,
            config.budget - cumulative_labeled,
            len(datamodule.unlabeled),
        )
        logger.info(f"\nStep 2: Selecting {num_to_select} most uncertain samples...")

        selected_indices = select_samples(
            unlabeled_data=unlabeled_data,
            training_data=training_data,
            model=model,
            num_samples=num_to_select,
            metric=config.uncertainty_metric,
            dataset_type=config.dataset_type,
            device=device,
            logger=logger,
        )

        if not selected_indices:
            logger.warning("No samples selected. Ending active learning loop.")
            break

        log_dataset_state(
            logger,
            iteration=iteration,
            datamodule=datamodule,
            stage="AFTER_SELECTION",
            extra_info={
                "Num selected": len(selected_indices),
                "Budget remaining": config.budget - cumulative_labeled - len(selected_indices),
            },
        )

        # Step 3: Move samples to training set
        logger.info(f"\nStep 3: Moving {len(selected_indices)} samples to training set...")
        datamodule.active_learning_add_label_data(indices_to_label=selected_indices)
        cumulative_labeled += len(selected_indices)

        # Validate dataset integrity
        validate_dataset_state(datamodule, total_samples, logger)

        log_dataset_state(
            logger,
            iteration=iteration,
            datamodule=datamodule,
            stage="AFTER_TRANSFER",
            extra_info={
                "Cumulative labeled": cumulative_labeled,
                "Progress": f"{cumulative_labeled}/{config.budget} ({100*cumulative_labeled/config.budget:.1f}%)",
                "Budget remaining": config.budget - cumulative_labeled,
            },
        )

        # Reinitialize model for next iteration
        del model
        model = get_model(config)
        model.to(device)

        iteration += 1

    # Final training
    logger.info(f"\n{'#'*60}")
    logger.info(f"# FINAL TRAINING ON ALL LABELED DATA")
    logger.info(f"{'#'*60}\n")

    model, final_metrics = train_samples(model, config, datamodule, iteration, logger)
    final_val_loss = final_metrics[0].get("val_loss", "N/A") if final_metrics else "N/A"

    eval_results.append(
        {
            "Iteration": iteration,
            "Batches_Labeled": cumulative_labeled // config.batch_size,
            "Samples_Labeled": cumulative_labeled,
            "Eval": final_metrics,
        }
    )

    log_dataset_state(
        logger,
        iteration=iteration,
        datamodule=datamodule,
        stage="FINAL",
        extra_info={
            "Final validation loss": (
                f"{final_val_loss:.4f}"
                if isinstance(final_val_loss, float)
                else str(final_val_loss)
            ),
            "Total iterations": iteration - 1,
            "Total labeled": cumulative_labeled,
        },
    )

    # Save results
    if config.plot_results:
        plot_path = plot_results(eval_results, config.log_dir, config.experiment_name)
        logger.info(f"Plot saved to: {plot_path}")

    log_results(config.log_dir, config.experiment_name, eval_results)

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Active Learning Experiment Completed!")
    logger.info(f"{'#'*60}")
    logger.info(f"Total samples labeled: {cumulative_labeled}/{config.budget}")
    logger.info(f"Total iterations: {iteration - 1}")
    logger.info(f"Final validation loss: {final_val_loss}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = args.parse_args()

    cfg = ActiveLearningConfig.from_yaml(args.config)

    try:
        main(cfg)
    except Exception as e:
        logging.error(f"Experiment failed with error: {e}", exc_info=True)
        raise
