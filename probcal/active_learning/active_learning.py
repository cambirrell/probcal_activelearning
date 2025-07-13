import argparse
import logging
import os.path
from datetime import datetime
from typing import Any

import math
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import yaml

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from probcal.enums import DatasetType
from probcal.enums import ImageDatasetName
from probcal.data_modules.probcal_datamodule import ProbcalDataModule
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.utils.configs import ActiveLearningConfig
from probcal.utils.experiment_utils import from_yaml
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.evaluation.calibration_evaluator import CalibrationEvaluator
from torch.utils.data import DataLoader


def train_samples(model: ProbabilisticRegressionNN, config:ActiveLearningConfig, datamodule: ProbcalDataModule):
    """
    Train the model on the sampled data.
    """
    fix_random_seed(config.random_seed)
    logger = CSVLogger(save_dir=config.log_dir, name=config.experiment_name)
    chkp_dir = config.chkp_dir / config.experiment_name 
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
    print("Point A1")
    trainer.fit(model=model, datamodule=datamodule)
    print("Point A2")
    val_metrics = trainer.validate(model=model, datamodule=datamodule)
    return model, val_metrics


def select_samples(
    unlabeled_data: DataLoader, training_data: DataLoader, model: ProbabilisticRegressionNN, num_samples: int, metric: str
):
    """
    Select samples from the unlabeled data based on the uncertainty metric.
    Must return unbatched data in form of a list[tuple[torch.Tensor, torch.Tensor]].
    unlabeled_data is a data loader right now, model is in lightening format,
    metric is string, and num_samples is the amount of data points to num_samples/batch_size is the target
    """
    evaluator = CalibrationEvaluator()
    # temporary assert statement, the check should be loading the config
    assert metric in ["cce"]
    if metric == "cce":
        uncertainty_scores, scored_batches = evaluator.compute__cce_active_learning(
            model, training_data, unlabeled_data
        )
        print("Point C2")
        print('k ', num_samples)
        print('scores: ', uncertainty_scores.shape)
        topk_indices = torch.topk(uncertainty_scores, k=num_samples).indices
        print("Point C3")
        highest_uncertainty_batches = [
            batch for i, batch in enumerate(unlabeled_data) if i in topk_indices.tolist()
        ]
        print("Point C4")
        data_to_label = []
        for x_batch, y_batch in highest_uncertainty_batches:
            # Unbind along the batch dimension (0) and pair up
            data_to_label.extend(list(zip(x_batch.unbind(0), y_batch.unbind(0))))
        print("Point C5")
    else:
        raise NotImplementedError # It is the plan to implement more uncertainty measures

    return data_to_label


def plot_results(results: Any, log_dir: str):
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
    plt.savefig(log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


def log_results(log_dir:str, eval_results: list) -> None:
    """
    Produces a YAML file with a heading for each training round.
    It includes eval metric, samples, and batches trained.
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
    log_path = f"active_learning_log_{timestamp}.yaml"
    with open(log_dir / log_path, "w") as f:
        yaml.dump(log_dict, f)
    print(f"Active learning log saved to {log_path}")

def main(config: ActiveLearningConfig) -> None:
    """
    Main function to run the active learning experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Add logging setup

    model = get_model(config)
    datamodule = get_datamodule(
        config.dataset_type, 
        config.dataset_path_or_spec, 
        config.batch_size, 
        config.num_workers
    )
    datamodule.setup("")
    datamodule.unlabeled_partion_setup(config.initial_labeled_partition)
    model.to(device)

    #To get the whole progress curve we will keep adding until we have trained on all the data
    #budget = config.budget
    eval_results = []
    num_labeled_batches = config.initial_labeled_partition
    unlabeled_data = datamodule.unlabeled_dataloader()
    # Lets see is dataloader has a len component to see if there is any data left
    while len(unlabeled_data) > 0:
        # Train the model on the sampled data
        model, val_metric = train_samples(model, config, datamodule)
        eval_results.append(
            {
                "Batches_Labeled": num_labeled_batches,
                "Samples_Labeled":num_labeled_batches * config.batch_size, 
                "Eval":val_metric
            }
            )
        print("Point C")
        # Select samples from the unlabeled data based on the uncertainty metric
        training_data = datamodule.train_dataloader()
        selected_samples = select_samples(unlabeled_data, training_data, model, config.sample_per_iteration, config.uncertainty_metric) 
        print("Point D")
        # update the training data with the selected samples
        training_data, unlabeled_data = datamodule.active_learning_add_labeled_data(
            data_to_label=selected_samples
        )
        print("Point E")
        # reinitialize the model to train on the new data
        del model
        model = get_model(config)
        model.to(device)


    model, val_metric = train_samples(model, config, datamodule)
    eval_results.append(
            {
                "Batches_Labeled": num_labeled_batches,
                "Samples_Labeled":num_labeled_batches * config.batch_size, 
                "Eval":val_metric
            }
            )
    if config.plot_results:
        plot_results(validation_data, config.log_dir)
    log_results(config.log_dir, eval_results)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    args = args.parse_args()
    cfg = ActiveLearningConfig.from_yaml(args.config)
    try:
        main(cfg)
    except Exception as e:
        logging.error(e)
