import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import lightning as L
import torch
import yaml

from probcal.enums import DatasetType
from probcal.evaluation.calibration_evaluator import CalibrationEvaluator
from probcal.evaluation.calibration_evaluator import CalibrationEvaluatorSettings
from probcal.evaluation.calibration_evaluator import CCESettings
from probcal.evaluation.calibration_evaluator import ECESettings
from probcal.evaluation.kernels import rbf_kernel
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


def active_learning_evaluation(
    model,
    datamodule,
    config: EvaluationConfig,
):
    """
    Evaluate the model on the active learning dataset.
    This function is a placeholder for the actual evaluation logic.
    """
    pass

def main(config_path: Path):

    config = EvaluationConfig.from_yaml(config_path)
    log_dir = config.log_dir / config.experiment_name
    if not log_dir.exists():
        os.makedirs(log_dir)

    datamodule = get_datamodule(
        config.dataset_type,
        config.dataset_path_or_spec,
        config.batch_size,
        config.num_workers,
    )

    model, initializer = get_model(config, return_initializer=True)
    try:
        model = initializer.load_from_checkpoint(config.model_ckpt_path)
    except Exception as e:
        print(f"{e} loading model from checkpoint. Trying to load from state dict...")
        model.load_state_dict(torch.load(config.model_ckpt_path))
        print("Model loaded from state dict.")

    evaluator = L.Trainer(
        accelerator=config.accelerator_type.value,
        enable_model_summary=False,
        logger=False,
        devices=1,
        num_nodes=1,
    )
    metrics: dict = evaluator.test(model=model, datamodule=datamodule)[0]
    metrics = {k: float(v) for k, v in metrics.items()}

    if config.dataset_type == DatasetType.TABULAR and config.input_dim == 1:
        x_vals = torch.cat([x for x, _ in datamodule.test_dataloader()], dim=0)
        gamma = (1 / (2 * x_vals.var())).item()
        cce_input_kernel = partial(rbf_kernel, gamma=gamma)
    else:
        cce_input_kernel = "polynomial"

    # TODO: For pure evaluation, we should use the test split for S. Separate file for forecasting error?
    use_val_split_for_S = True

    cce_settings = CCESettings(
        use_val_split_for_S=use_val_split_for_S,
        num_trials=config.cce_num_trials,
        num_mc_samples=config.cce_num_mc_samples,
        input_kernel=cce_input_kernel,
        output_kernel=config.cce_output_kernel,
        lmbda=config.cce_lambda,
    )
    ece_settings = ECESettings(
        num_bins=config.ece_bins,
        weights=config.ece_weights,
        alpha=config.ece_alpha,
    )
    prob_eval_settings = CalibrationEvaluatorSettings(
        dataset_type=config.dataset_type,
        device=torch.device("cuda" if config.accelerator_type.value == "gpu" else "cpu"),
        cce_settings=cce_settings,
        ece_settings=ece_settings,
        num_bootstrap_samples=10,
    )
    calib_evaluator = CalibrationEvaluator(settings=prob_eval_settings)
    print("Evaluating calibration...")
    results = calib_evaluator(model=model, data_module=datamodule)

    metrics.update(
        mean_cce_bar=results.cce.mean_cce_bar,
        std_cce_bar=results.cce.std_cce_bar,
        mean_ece=results.ece.mean_ece,
        std_ece=results.ece.std_ece,
    )
    with open(log_dir / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)
    results.save(log_dir / "calibration_results.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to evaluation config.yaml.")
    args = parser.parse_args()
    main(config_path=Path(args.config))
