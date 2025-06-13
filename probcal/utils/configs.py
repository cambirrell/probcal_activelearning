from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from probcal.enums import AcceleratorType
from probcal.enums import BetaSchedulerType
from probcal.enums import DatasetType
from probcal.enums import HeadType
from probcal.enums import ImageDatasetName
from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.utils.generic_utils import get_yaml
from probcal.utils.generic_utils import to_snake_case


class BaseConfig(object):
    def __init__(
        self,
        experiment_name: str,
        head_type: HeadType,
        dataset_type: DatasetType,
        dataset_path_or_spec: Path | ImageDatasetName,
        source_dict: dict,
        input_dim: int,
        hidden_dim: int,
        batch_size: int,
        num_workers: int,
        accelerator_type: AcceleratorType,
        log_dir: Path,
    ):
        self.experiment_name = experiment_name
        self.head_type = head_type
        self.dataset_type = dataset_type
        self.dataset_path_or_spec = dataset_path_or_spec
        self.source_dict = source_dict
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accelerator_type = accelerator_type
        self.log_dir = log_dir

    def to_yaml(self, filepath: str | Path):
        """Save this config as a .yaml file at the given filepath.

        Args:
            filepath (str | Path): The filepath to save this config at.
        """
        with open(filepath, "w") as f:
            yaml.dump(self.source_dict, f)

    @staticmethod
    def from_yaml(config_path: str | Path) -> TrainingConfig:
        """Factory method to construct an TrainingConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TrainingConfig: The specified config.
        """
        raise NotImplementedError("This method should be implemented in a subclass of BaseConfig.")

    @staticmethod
    def get_dataset_path_or_spec(dataset_cfg: dict):
        if "path" in dataset_cfg:
            return Path(dataset_cfg["path"])
        else:
            type = DatasetType(dataset_cfg["type"])
            spec = dataset_cfg["spec"]
            if type == DatasetType.TABULAR:
                return Path(dataset_cfg["path"])
            if type == DatasetType.IMAGE:
                return ImageDatasetName(spec)
            else:
                return Path(spec)


class TrainingConfig(BaseConfig):
    """Class with configuration options for training a model.

    Attributes:
        experiment_name (str): The name of the training run (used for identifying chkp weights / eval logs), automatically cast to snake case.
        head_type (HeadType): The output head to use in the neural network, e.g. "gaussian", "poisson", etc.
        chkp_dir (Path): Directory to checkpoint model weights in.
        chkp_freq (int): Number of epochs to wait in between checkpointing model weights.
        batch_size (int): The batch size to train with.
        num_epochs (int): The number of epochs through the data to complete during training.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        dataset_type (DatasetType): Type of dataset to use in this experiment.
        dataset_path_or_spec (Path | ImageDatasetName): If specifying a tabular dataset, path to the dataset .npz file to use. Otherwise, the name of the dataset.
        num_trials (int): Number of trials to run for this experiment.
        log_dir (Path): Directory to log results to.
        source_dict (dict): Dictionary from which config was constructed.
        input_dim (int): The input dim of the data (used to construct the MLP). Defaults to 1.
        hidden_dim (int, optional): Feature dimension used in the model (before feeding the representation to the output head). Defaults to 64.
        precision (str | None, optional): String specifying desired floating point precision for training. Defaults to None.
        random_seed (int | None, optional): If specified, the random seed to use for reproducibility. Defaults to None.
    """

    def __init__(
        self,
        experiment_name: str,
        accelerator_type: AcceleratorType,
        head_type: HeadType,
        chkp_dir: Path,
        chkp_freq: int,
        batch_size: int,
        num_workers: int,
        num_epochs: int,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None,
        lr_scheduler_kwargs: dict | None,
        beta_scheduler_type: BetaSchedulerType | None,
        beta_scheduler_kwargs: dict | None,
        dataset_type: DatasetType,
        dataset_path_or_spec: Path | ImageDatasetName,
        num_trials: int,
        log_dir: Path,
        source_dict: dict,
        input_dim: int = 1,
        hidden_dim: int = 64,
        precision: str | None = None,
        random_seed: int | None = None,
    ):
        super(TrainingConfig, self).__init__(
            experiment_name=experiment_name,
            head_type=head_type,
            dataset_type=dataset_type,
            source_dict=source_dict,
            dataset_path_or_spec=dataset_path_or_spec,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator_type=accelerator_type,
            log_dir=log_dir,
        )
        self.chkp_dir = chkp_dir
        self.chkp_freq = chkp_freq
        self.num_epochs = num_epochs
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.beta_scheduler_type = beta_scheduler_type
        self.beta_scheduler_kwargs = beta_scheduler_kwargs
        self.num_trials = num_trials
        self.precision = precision
        self.random_seed = random_seed

    @staticmethod
    def from_yaml(config_path: str | Path) -> TrainingConfig:
        """Factory method to construct an TrainingConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TrainingConfig: The specified config.
        """
        config_dict = get_yaml(config_path)
        training_dict: dict = config_dict["training"]
        eval_dict: dict = config_dict["evaluation"]

        experiment_name = to_snake_case(config_dict["experiment_name"])
        accelerator_type = AcceleratorType(training_dict["accelerator"])
        head_type = HeadType(config_dict["head_type"])
        chkp_dir = Path(training_dict["chkp_dir"])
        chkp_freq = training_dict["chkp_freq"]
        batch_size = training_dict["batch_size"]
        num_workers = training_dict.get("num_workers", 8)
        num_epochs = training_dict["num_epochs"]
        precision = training_dict.get("precision")
        optim_type = OptimizerType(training_dict["optimizer"]["type"])
        optim_kwargs = training_dict["optimizer"]["kwargs"]

        if "lr_scheduler" in training_dict:
            lr_scheduler_type = LRSchedulerType(training_dict["lr_scheduler"]["type"])
            lr_scheduler_kwargs = training_dict["lr_scheduler"]["kwargs"]
        else:
            lr_scheduler_type = None
            lr_scheduler_kwargs = None

        if "beta_scheduler" in training_dict:
            beta_scheduler_type = BetaSchedulerType(training_dict["beta_scheduler"]["type"])
            beta_scheduler_kwargs = training_dict["beta_scheduler"]["kwargs"]
            if beta_scheduler_kwargs.get("last_epoch", None) == -1:
                beta_scheduler_kwargs["last_epoch"] = num_epochs
        else:
            beta_scheduler_type = None
            beta_scheduler_kwargs = None

        dataset_type = DatasetType(config_dict["dataset"]["type"])
        dataset_path_or_spec = TrainingConfig.get_dataset_path_or_spec(config_dict["dataset"])

        num_trials = eval_dict["num_trials"]
        log_dir = Path(eval_dict["log_dir"])
        input_dim = config_dict["dataset"].get("input_dim", 1)
        hidden_dim = config_dict.get("hidden_dim", 64)
        random_seed = config_dict.get("random_seed")

        return TrainingConfig(
            experiment_name=experiment_name,
            accelerator_type=accelerator_type,
            head_type=head_type,
            chkp_dir=chkp_dir,
            chkp_freq=chkp_freq,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            beta_scheduler_type=beta_scheduler_type,
            beta_scheduler_kwargs=beta_scheduler_kwargs,
            dataset_type=dataset_type,
            dataset_path_or_spec=dataset_path_or_spec,
            num_trials=num_trials,
            log_dir=log_dir,
            source_dict=config_dict,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            precision=precision,
            random_seed=random_seed,
        )


class EvaluationConfig(BaseConfig):
    """Class with configuration options for testing a model."""

    def __init__(
        self,
        experiment_name: str,
        head_type: HeadType,
        dataset_type: DatasetType,
        source_dict: dict,
        dataset_path_or_spec: Path | ImageDatasetName,
        input_dim: int,
        hidden_dim: int,
        batch_size: int,
        num_workers: int,
        accelerator_type: AcceleratorType,
        log_dir: Path,
        model_ckpt_path: Path,
        cce_use_val_split_for_S: bool,
        cce_num_trials: int,
        cce_output_kernel: Literal["rbf", "laplacian"],
        cce_lambda: float,
        cce_num_mc_samples: int,
        ece_bins: int,
        ece_weights: Literal["uniform", "frequency"],
        ece_alpha: float,
    ):
        super(EvaluationConfig, self).__init__(
            experiment_name=experiment_name,
            head_type=head_type,
            dataset_type=dataset_type,
            source_dict=source_dict,
            dataset_path_or_spec=dataset_path_or_spec,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator_type=accelerator_type,
            log_dir=log_dir,
        )
        self.model_ckpt_path = model_ckpt_path
        self.cce_use_val_split_for_S = cce_use_val_split_for_S
        self.cce_num_trials = cce_num_trials
        self.cce_output_kernel = cce_output_kernel
        self.cce_lambda = cce_lambda
        self.cce_num_mc_samples = cce_num_mc_samples
        self.ece_bins = ece_bins
        self.ece_weights = ece_weights
        self.ece_alpha = ece_alpha

    @staticmethod
    def from_yaml(config_path: str | Path) -> EvaluationConfig:
        """Factory method to construct an TestConfig from a .yaml file.

        Args:
            config_path (str | Path): Path to the .yaml file with config options.

        Returns:
            TestConfig: The specified config.
        """
        config_dict = get_yaml(config_path)
        experiment_name = to_snake_case(config_dict["experiment_name"])

        # Required arguments.
        head_type = HeadType(config_dict["head_type"])
        dataset_type = DatasetType(config_dict["dataset"]["type"])

        log_dir = Path(config_dict["log_dir"])
        model_ckpt_path = Path(config_dict["model_ckpt_path"])
        dataset_path_or_spec = TrainingConfig.get_dataset_path_or_spec(config_dict["dataset"])

        # Optional arguments.
        input_dim = config_dict["dataset"].get("input_dim", 1)
        hidden_dim = config_dict.get("hidden_dim", 64)
        batch_size = config_dict.get("batch_size", 1)
        num_workers = config_dict.get("num_workers", 0)
        accelerator_type = AcceleratorType(config_dict.get("accelerator", "cpu"))

        cce_settings: dict = config_dict.get("cce_settings", {})
        cce_use_val_split_for_S = cce_settings.get("use_val_split_for_S", False)
        cce_num_trials = cce_settings.get("num_trials", 5)
        cce_output_kernel = cce_settings.get("output_kernel", "rbf")
        if cce_output_kernel not in ("rbf", "laplacian"):
            raise ValueError("output_kernel for CCE must be either 'rbf' or 'laplacian'.")
        cce_lambda = cce_settings.get("lambda", 0.1)
        cce_num_mc_samples = cce_settings.get("num_mc_samples", 1)

        ece_settings: dict = config_dict.get("ece_settings", {})
        ece_bins = ece_settings.get("num_bins", 50)
        ece_weights = ece_settings.get("weights", "frequency")
        if ece_weights not in ("frequency", "uniform"):
            raise ValueError("weights for ECE must be either 'frequency' or 'uniform'.")
        ece_alpha = ece_settings.get("alpha", 1.0)

        return EvaluationConfig(
            experiment_name=experiment_name,
            head_type=head_type,
            dataset_type=dataset_type,
            source_dict=config_dict,
            dataset_path_or_spec=dataset_path_or_spec,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator_type=accelerator_type,
            log_dir=log_dir,
            model_ckpt_path=model_ckpt_path,
            cce_use_val_split_for_S=cce_use_val_split_for_S,
            cce_num_trials=cce_num_trials,
            cce_output_kernel=cce_output_kernel,
            cce_lambda=cce_lambda,
            cce_num_mc_samples=cce_num_mc_samples,
            ece_bins=ece_bins,
            ece_weights=ece_weights,
            ece_alpha=ece_alpha,
        )
    


class ActiveLearningConfig(BaseConfig):
    def __init__(self, experiment_name, head_type, dataset_type, dataset_path_or_spec, source_dict, input_dim, hidden_dim, batch_size, num_workers, accelerator_type, log_dir):
        super().__init__(experiment_name, head_type, dataset_type, dataset_path_or_spec, source_dict, input_dim, hidden_dim, batch_size, num_workers, accelerator_type, log_dir)
    #TODO: Fill this out

    
