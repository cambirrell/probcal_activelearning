import argparse
import logging
import os.path
from datetime import datetime
from typing import Any, List

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
    # This function remains unchanged
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
    trainer.fit(model=model, datamodule=datamodule)
    val_metrics = trainer.validate(model=model, datamodule=datamodule)
    return model, val_metrics


def select_samples(
    unlabeled_data: DataLoader, training_data: DataLoader, model: ProbabilisticRegressionNN, num_batches_to_sample: int, metric: str
) -> List[int]:
    """
    Selects samples from the unlabeled data pool based on an uncertainty metric.
    
    This function now returns a list of the original dataset indices for the
    highest-uncertainty samples.
    """
    evaluator = CalibrationEvaluator()
    assert metric in ["cce"], "Currently only 'cce' metric is supported."
    if metric == "cce":
        # This function returns a score for each batch in the dataloader
        uncertainty_scores, _ = evaluator.compute__cce_active_learning(
            model, training_data, unlabeled_data
        )

        # The unlabeled_dataloader now yields (data, target, original_index).
        # We extract all the original indices from the dataloader's batches.
        print(f"returning indices from unlabeled data {unlabeled_data.dataset.return_index}")
        all_original_indices_in_batches = [batch[2] for batch in unlabeled_data]

        # Get the indices of the top-k batches with the highest uncertainty
        num_to_sample = min(num_batches_to_sample, len(uncertainty_scores))
        topk_batch_indices = torch.topk(uncertainty_scores, k=num_to_sample).indices

        # Collect all the original sample indices from within those top batches
        selected_sample_indices = []
        for batch_idx in topk_batch_indices:
            # batch[2] is a tensor of original indices for that batch
            selected_sample_indices.extend(all_original_indices_in_batches[batch_idx].tolist())
        
        return selected_sample_indices
    else:
        raise NotImplementedError


def plot_results(results: Any, log_dir: str):
    """
    Plot the results of the model as more data is sampled.
    """
    # This function remains unchanged
    plt.figure(figsize=(10, 5))
    plt.plot(results)
    plt.title("Model Performance Over Time")
    plt.xlabel("Number of Samples")
    plt.ylabel("Performance Metric")
    plt.grid()
    plt.savefig(log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


def log_results(log_dir:str, eval_results: list) -> None:
    """
    Produces a YAML file with a heading for each training round.
    """
    # This function remains unchanged
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

    model = get_model(config)
    datamodule = get_datamodule(
        config.dataset_type, 
        config.dataset_path_or_spec, 
        config.batch_size, 
        config.num_workers
    )
    datamodule.setup("")
    datamodule.unlabeled_partion_setup(config.initial_labeled_partition * config.batch_size)
    model.to(device)

    eval_results = []
    num_labeled_batches = config.initial_labeled_partition
    
    unlabeled_data = datamodule.unlabeled_dataloader()
    while len(datamodule.unlabeled) > 0:
        print(f"\n--- Starting new AL round. Labeled: {len(datamodule.train)}, Unlabeled: {len(datamodule.unlabeled)} ---")
        
        model, val_metric = train_samples(model, config, datamodule)
        
        eval_results.append(
            {
                "Batches_Labeled": num_labeled_batches,
                "Samples_Labeled": len(datamodule.train), 
                "Eval": val_metric
            }
        )
        
        training_data = datamodule.train_dataloader()
        unlabeled_data = datamodule.unlabeled_dataloader()

        if len(unlabeled_data) == 0:
            break

        # Select the *indices* of samples to label next
        selected_indices = select_samples(
            unlabeled_data, 
            training_data, 
            model, 
            config.sample_per_iteration, 
            config.uncertainty_metric
        ) 

        if not selected_indices:
            print("No new samples selected to label. Ending active learning loop.")
            break
        
        print(f"Labeling {len(selected_indices)} new samples identified by uncertainty...")
        
        # Update the datasets by passing the list of indices
        datamodule.active_learning_add_label_data(
            indices_to_label=selected_indices
        )
        
        num_labeled_batches += config.sample_per_iteration

        # Reinitialize the model to train on the new data from scratch
        del model
        model = get_model(config)
        model.to(device)

    print("\n--- All data has been labeled. Performing final training run. ---")
    model, val_metric = train_samples(model, config, datamodule)
    eval_results.append(
        {
            "Batches_Labeled": num_labeled_batches,
            "Samples_Labeled": len(datamodule.train),
            "Eval": val_metric
        }
    )
    
    if config.plot_results and eval_results:
        # Assuming we plot based on one of the validation metrics
        performance_curve = [r['Eval'][0]['val/mse'] for r in eval_results]
        plot_results(performance_curve, config.log_dir)
        
    log_results(config.log_dir, eval_results)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args = args.parse_args()
    cfg = ActiveLearningConfig.from_yaml(args.config)
    try:
        main(cfg)
    except Exception as e:
        logging.exception(e) # Use logging.exception to include stack trace