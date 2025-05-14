import argparse
import logging
import os.path
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from probcal.enums import DatasetType
from probcal.enums import ImageDatasetName
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import from_yaml
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model

def sample_data(num_samples: int, num_classes: int):
    """
    Sample data from the dataset.
    """
    pass

def train_samples(num_samples: int, num_classes: int):
    """
    Train the model on the sampled data.
    """
    pass

def eval_model(validation_data: Any):
    """
    Evaluate the model on the validation data.
    """
    pass

def select_samples(unlabeled_data: Any, num_samples: int, metric: str):
    """
    Select samples from the unlabeled data based on the uncertainty metric.
    """
    pass

def uncertainty_estimation(model: Any, unlabeled_data: Any, num_samples: int, metric: str):
    """
    Estimate the uncertainty of the model on the data.
    """
    pass

def plot_results(results: Any):
    """
    Plot the results of the model as more data is sampled.
    """
    pass


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
    model.to(device)

    budget = config.budget
    eval_results = []
    # make initial sample, and get a validation set
    training_data, validation_data, unlabeled_data = sample_data(config.validation_size, config.num_classes)
    while budget > 0:
        # Train the model on the sampled data
        train_samples(training_data, config.num_classes)

        # Evaluate the model on the validation data
        eval_results.append(eval_model(validation_data))

        # Select samples from the unlabeled data based on the uncertainty metric
        selected_samples = select_samples(unlabeled_data, config.num_samples, config.metric)

        #update the training data with the selected samples
        training_data = torch.cat((training_data, selected_samples), dim=0)        

        # Remove the selected samples from the unlabeled data
        unlabeled_data = unlabeled_data[~unlabeled_data.isin(selected_samples)]

        budget -= config.num_samples
    train_samples(training_data, config.num_classes)
    eval_results.append(eval_model(validation_data))
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
