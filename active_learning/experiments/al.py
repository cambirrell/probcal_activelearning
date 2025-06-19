import argparse
import logging
import os.path
from datetime import datetime
from typing import Any

import math
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from probcal.enums import DatasetType
from probcal.enums import ImageDatasetName
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.utils.configs import ActiveLearningConfig
from probcal.utils.experiment_utils import from_yaml
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.evaluation.calibration_evaluator import CalibrationEvaluator


def train_samples(model: Any, config, training_data: Any, validation_data: Any, num_classes: int):
    """
    Train the model on the sampled data.
    """
    fix_random_seed(config.random_seed)
    logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)
    chkp_dir = config.chkp_dir / config.experiment_name / f"version_{i}"
    chkp_callbacks = get_chkp_callbacks(chkp_dir, config.chkp_freq)
    trainer = L.Trainer(
            accelerator=config.accelerator_type.value,
            min_epochs=config.num_epochs,
            max_epochs=config.num_epochs,
            log_every_n_steps=5,
            check_val_every_n_epoch=math.ceil(config.num_epochs / 200),
            enable_model_summary=False,
            callbacks=chkp_callbacks,
            logger=logger,
            precision=config.precision,
        )
    trainer.fit(model=model, datamodule=training_data)
    val_metrics = trainer.validate(model=model, dataloaders=validation_data)
    return model, val_metrics
    

def select_samples(unlabeled_data: Any, training_data: Any, model: Any, num_samples: int, metric: str):
    """
    Select samples from the unlabeled data based on the uncertainty metric.
    Must return unbatched data in form of a list[tuple[torch.Tensor, torch.Tensor]].
    unlabeled_data is a data loader right now, model is in lightening format,
    metric is string, and num_samples is the amount of data points to num_samples/batch_size is the target
    """
    evaluator = CalibrationEvaluator()
    assert metric in ["cce"]
    if metric == "cce":
        grid_loader = None
        uncertainty_scores, scored_batches = evaluator.compute__cce_active_learning(model, grid_loader, training_data, unlabeled_data)
        topk_indices = torch.topk(uncertainty_scores, k=num_samples).indices
        highest_uncertainty_batches = [batch for i, batch in enumerate(unlabeled_data) if i in topk_indices.tolist()]
        data_to_label = []
        for x_batch, y_batch in highest_uncertainty_batches:
            # Unbind along the batch dimension (0) and pair up
            data_to_label.extend(list(zip(x_batch.unbind(0), y_batch.unbind(0))))
    else:
        raise NotImplementedError
    
    return data_to_label

    

def plot_results(results: Any):
    """
    Plot the results of the model as more data is sampled.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(results)
    plt.title("Model Performance Over Time")
    plt.xlabel("Number of Samples")
    plt.ylabel("Performance Metric")
    plt.grid()
    # TODO: Add more details to the plot, like confidence intervals or error bars
    # TODO: Also give a more descriptive name and save the plot in a specific location
    plt.savefig(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


# TODO: Add a config class for active learning
def main(config: ActiveLearningConfig) -> None:
    """
    Main function to run the active learning experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Add logging setup

    model = get_model(
        config.model_type,
        config.model_params,
        config.model_name,
        config.model_path,
    )
    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
    )
    datamodule.setup()
    # TODO: See if this generated if logic is needed
    if config.dataset_type == DatasetType.IMAGE and config.image_dataset_name is not None:
        datamodule.set_image_dataset(config.image_dataset_name)
    elif config.dataset_type == DatasetType.TABULAR and config.input_dim == 1:
        datamodule.set_tabular_dataset(config.input_dim)
    else:
        raise ValueError("Unsupported dataset type or configuration.")
    datamodule.unlabeled_partion_setup(config.partition_size)

    model.to(device)

    budget = config.budget
    eval_results = []
    # make initial sample, and get a validation set
    # Each of these should be a datamodule type
    #the splits are also predetermined and split, so we just need to mask part of the training data and split it into unlabeled 
    training_data = datamodule.train_dataloader()
    validation_data = datamodule.val_dataloader()
    unlabeled_data = datamodule.unlabeled_dataloader()
    while budget > 0:
        # Train the model on the sampled data
        model, val_metric = train_samples(training_data, config.num_classes, model)
        eval_results.append(val_metric)
        # Select samples from the unlabeled data based on the uncertainty metric
        selected_samples = select_samples(unlabeled_data, config.num_samples, config.metric)

        #update the training data with the selected samples
        training_data, unlabeled_data = datamodule.active_learning_add_labeled_data(data_to_label=selected_samples)     

        budget -= config.num_samples

        # reinitialize the model to train on the new data
        model = get_model(
            config.model_type,
            config.model_params,
            config.model_name,
            config.model_path,
        )
    model, val_metric = train_samples(training_data, config.num_classes)
    eval_results.append(val_metric)
    if config.plot_results:
        plot_results(validation_data)

    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    args = args.parse_args()
    cfg = from_yaml(args.config)
    try:
        main(cfg)
    except Exception as e:
        logging.error(e)
