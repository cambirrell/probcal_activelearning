from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import TypeAlias

import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from open_clip import CLIP
from scipy.interpolate import griddata
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from probcal.data_modules.probcal_datamodule import ProbcalDataModule
from probcal.enums import DatasetType
from probcal.evaluation.kernels import laplacian_kernel
from probcal.evaluation.kernels import polynomial_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.evaluation.metrics import compute_regression_ece
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN


@dataclass
class CCEResults:
    expected_values: torch.Tensor  # (n,)
    stdevs: torch.Tensor  # (n,)
    quantiles: dict[str, torch.Tensor]  # Each is (n,)
    mean_cce_bar: float
    std_cce_bar: float


@dataclass
class ECEResults:
    mean_ece: float
    std_ece: float
    quantiles: dict[str, float]


@dataclass
class CalibrationResults:
    targets: np.ndarray
    input_grid_2d: np.ndarray
    cce: CCEResults
    ece: ECEResults

    def save(self, filepath: str | Path):
        """Save calibration results to the given filepath.

        Args:
            filepath (str | Path): Path to save results to. Must end with .pt

        Raises:
            ValueError: If the filepath does not end with .pt
        """
        if not str(filepath).endswith(".pt"):
            raise ValueError("Filepath must have a .pt extension.")
        save_dict = {
            "input_grid_2d": self.input_grid_2d,
            "targets": self.targets,
            "cce": asdict(self.cce),
            "ece": asdict(self.ece),
        }
        torch.save(save_dict, filepath)

    @staticmethod
    def load(filepath: str | Path) -> CalibrationResults:
        """Load calibration results from the given filepath.

        Args:
            filepath (str | Path): Path to load results from.

        Returns:
            CalibrationResults: Python representation of the results.
        """
        data = torch.load(filepath)
        ece_results = ECEResults(**data["ece"])
        cce_results = CCEResults(**data["cce"])
        return CalibrationResults(
            input_grid_2d=data["input_grid_2d"],
            targets=data["targets"],
            cce=cce_results,
            ece=ece_results,
        )


KernelFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class CCESettings:
    use_val_split_for_S: bool = False
    num_trials: int = 5
    num_mc_samples: int = 1
    input_kernel: Literal["polynomial"] | KernelFunction = "polynomial"
    output_kernel: Literal["rbf", "laplacian"] | KernelFunction = "rbf"
    lmbda: float = 0.1


@dataclass
class ECESettings:
    num_bins: int = 50
    weights: Literal["uniform", "frequency"] = "frequency"
    alpha: float = 1.0


@dataclass
class CalibrationEvaluatorSettings:
    dataset_type: DatasetType = DatasetType.IMAGE
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_bootstrap_samples: int = 10
    cce_settings: CCESettings = CCESettings()
    ece_settings: ECESettings = ECESettings()


class CalibrationEvaluator:
    """Helper object to evaluate the probabilistic fit of a neural net."""

    def __init__(self, settings: CalibrationEvaluatorSettings = CalibrationEvaluatorSettings()):
        self.settings = settings
        self.device = settings.device
        self.image_preprocess = Resize(size=224)
        self._clip_model = None
        self._tokenizer = None
        self._grid_cache_path = Path("/tmp/cce_grid_cache.pt")
        self._cleanup()

    @torch.inference_mode()
    def __call__(
        self, model: ProbabilisticRegressionNN, data_module: ProbcalDataModule
    ) -> CalibrationResults:

        model.to(self.device)
        data_module.prepare_data()
        data_module.setup("")

        b = self.settings.num_bootstrap_samples
        t = self.settings.cce_settings.num_trials
        n = len(data_module.test)

        ece_draws = torch.zeros(b)
        cce_draws = torch.zeros(b, t, n)
        for i in tqdm(range(b), desc="Computing metrics on bootstraps"):

            # We first define our iterator over a bootstrap sample of S.
            if self.settings.cce_settings.use_val_split_for_S:
                data_module.set_bootstrap_indices("val")
                sample_loader = data_module.val_dataloader()
            else:
                data_module.set_bootstrap_indices("test")
                sample_loader = data_module.test_dataloader()

            # We always compute CCE over the same test points. The bootstrap only modifies how we define S.
            data_module.clear_bootstrap_indices("test")
            grid_loader = data_module.test_dataloader()

            # The ECE needs its own bootstrap sample.
            data_module.set_bootstrap_indices("test")
            ece_loader = data_module.test_dataloader()

            for j in tqdm(range(t), desc=f"Running {t} CCE computations", leave=False):
                cce_vals, grid, targets = self.compute_cce(
                    model=model,
                    grid_loader=grid_loader,
                    sample_loader=sample_loader,
                    return_grid=True,
                    return_targets=True,
                )

                cce_draws[i, j] = cce_vals

                # We only need to save the input grid / regression targets once.
                if i == 0 and j == 0:
                    if self.settings.dataset_type == DatasetType.TABULAR:
                        grid_2d = np.array([])
                    else:
                        grid_2d = TSNE(n_jobs=-1).fit_transform(grid.detach().cpu().numpy())
                    regression_targets = targets.detach().cpu().numpy()

            ece_draws[i] = self.compute_ece(model, ece_loader)

            flattened_draws = cce_draws.flatten(0, 1)
            expected_values = flattened_draws.mean(dim=0)
            stdevs = flattened_draws.std(dim=0)
            q_vals = ("0.025", "0.05", "0.1", "0.9", "0.95", "0.975")
            cce_quantiles = {
                q: torch.quantile(flattened_draws, torch.tensor(float(q)), dim=0) for q in q_vals
            }
            mean_cce_bar = cce_draws.mean(dim=2).flatten().mean().item()
            std_cce_bar = cce_draws.mean(dim=2).flatten().std().item()
            cce_results = CCEResults(
                expected_values=expected_values,
                stdevs=stdevs,
                quantiles=cce_quantiles,
                mean_cce_bar=mean_cce_bar,
                std_cce_bar=std_cce_bar,
            )

            ece_quantiles = {q: torch.quantile(ece_draws, torch.tensor(float(q))) for q in q_vals}
            ece_results = ECEResults(
                mean_ece=ece_draws.mean().item(),
                std_ece=ece_draws.std().item(),
                quantiles=ece_quantiles,
            )

        self._cleanup()

        return CalibrationResults(
            input_grid_2d=grid_2d,
            targets=regression_targets,
            cce=cce_results,
            ece=ece_results,
        )

    def compute_cce(
        self,
        model: ProbabilisticRegressionNN,
        grid_loader: DataLoader,
        sample_loader: DataLoader,
        return_grid: bool = False,
        return_targets: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Compute the CCE between the given model and data samples.

        Args:
            model (ProbabilisticRegressionNN): Probabilistic regression model to compute the CCE for.
            grid_loader (DataLoader): DataLoader with the data inputs to compute CCE over.
            sample_loader (DataLoader): DataLoader with the data inputs/outputs that define S (from which we form our two samples).
            return_grid (bool, optional): Whether/not to return the grid of values the CCE was computed over. Defaults to False.
            return_targets (bool, optional): Whether/not to return the regression targets the CCE was computed against. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The computed CCE values, along with the grid of inputs these values correspond to (if return_grid is True) and the regression targets (if return_targets is True).
        """
        x, y, x_prime, y_prime = self._get_samples_for_mcmd(model, sample_loader)
        grid = self._load_or_compute_grid(grid_loader)
        x_kernel, y_kernel = self._get_kernel_functions(y)
        cce_vals = compute_mcmd_torch(
            grid=grid,
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            x_kernel=x_kernel,
            y_kernel=y_kernel,
            lmbda=self.settings.cce_settings.lmbda,
        )
        return_obj = [cce_vals]
        if return_grid:
            return_obj.append(grid)
        if return_targets:
            return_obj.append(y)
        if len(return_obj) == 1:
            return return_obj[0]
        else:
            return tuple(return_obj)
        

    def compute__cce_active_learning(
            self,
            model: ProbabilisticRegressionNN,
            grid_loader: DataLoader,
            sample_loader: DataLoader,
            unlabeled_sample_loader: DataLoader,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute the CCE between the given model and data samples.

            Args:
                model (ProbabilisticRegressionNN): Probabilistic regression model to compute the CCE for.
                grid_loader (DataLoader): DataLoader with the data inputs to compute CCE over.
                sample_loader (DataLoader): DataLoader with the data inputs/outputs that define S for x and y.
                unlabeled_sample_loader (DataLoader): DataLoader that we will put x through model to form x prime and y prime samples.
            Returns:
                tuple[torch.Tensor, torch.Tensor] 
            """
            pass

    def compute_ece(self, model: ProbabilisticRegressionNN, data_loader: DataLoader) -> float:
        """Compute the regression ECE of the given model over the dataset spanned by the data loader.

        Args:
            model (ProbabilisticRegressionNN): Probabilistic regression model to compute the ECE for.
            data_loader (DataLoader): DataLoader with the test data to compute ECE over.

        Returns:
            float: The regression ECE.
        """
        all_outputs = []
        all_targets = []
        for inputs, targets in data_loader:
            all_outputs.append(model.predict(inputs.to(self.device)))
            all_targets.append(targets.to(self.device))

        all_targets = torch.cat(all_targets).detach().cpu().numpy()
        all_outputs = torch.cat(all_outputs, dim=0)
        posterior_predictive = model.predictive_dist(all_outputs)

        ece = compute_regression_ece(
            y_true=all_targets,
            posterior_predictive=posterior_predictive,
            num_bins=self.settings.ece_settings.num_bins,
            weights=self.settings.ece_settings.weights,
            alpha=self.settings.ece_settings.alpha,
        )
        return ece

    @staticmethod
    def plot_cce_results(
        results: CalibrationResults,
        gridsize: int = 100,
        show: bool = False,
    ) -> plt.Figure:
        """Given a set of evaluation results and an existing axes, plot the CCE values against their 2d input projections on a hexbin grid. Also depict their bootstrapped uncertainties.

        Args:
            results (CalibrationResults): Results from a CalibrationEvaluator.
            ax (plt.Axes): The axes to draw the plot on.
            gridsize (int, optional): Determines the granularity of the contour plot (higher numbers are more granular). Defaults to 100.
            show (bool, optional): Whether/not to show the resultant figure with plt.show(). Defaults to False.

        Returns:
            Figure: Matplotlib figure with the visualized CCE values.
        """
        cce_means = results.cce.expected_values
        cce_stdevs = results.cce.stdevs
        input_grid_2d = results.input_grid_2d

        fig, axs = plt.subplots(1, 3, figsize=(11, 3))
        axs: Sequence[plt.Axes]
        grid_x, grid_y = np.mgrid[
            min(input_grid_2d[:, 0]) : max(input_grid_2d[:, 0]) : gridsize * 1j,
            min(input_grid_2d[:, 1]) : max(input_grid_2d[:, 1]) : gridsize * 1j,
        ]

        axs[0].set_title("Data")
        grid_data = griddata(
            input_grid_2d,
            results.targets,
            (grid_x, grid_y),
            method="linear",
        )
        mappable_0 = axs[0].contourf(grid_x, grid_y, grid_data, levels=5, cmap="viridis")
        fig.colorbar(mappable_0, ax=axs[0])

        axs[1].set_title("CCE (Mean)")
        grid_cce_means = griddata(input_grid_2d, cce_means, (grid_x, grid_y), method="linear")
        mappable_1 = axs[1].contourf(grid_x, grid_y, grid_cce_means, levels=5, cmap="viridis")
        fig.colorbar(mappable_1, ax=axs[1])

        axs[2].set_title("CCE (Std)")
        grid_cce_stdevs = griddata(input_grid_2d, cce_stdevs, (grid_x, grid_y), method="linear")
        mappable_2 = axs[2].contourf(grid_x, grid_y, grid_cce_stdevs, levels=5, cmap="viridis")
        fig.colorbar(mappable_2, ax=axs[2])

        fig.tight_layout()

        if show:
            plt.show()
        return fig

    def _get_samples_for_mcmd(
        self,
        model: ProbabilisticRegressionNN,
        sample_loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = []
        y = []
        x_prime = []
        y_prime = []
        loop = tqdm(sample_loader, desc="Drawing samples", leave=False)
        for inputs, targets in loop:
            inputs: torch.Tensor
            targets: torch.Tensor

            if self.settings.dataset_type == DatasetType.TABULAR:
                encoder = self._encode_tabular
            elif self.settings.dataset_type == DatasetType.IMAGE:
                encoder = self._encode_image
            else:
                encoder = self._encode_text

            encoded_inputs = encoder(inputs.to(self.device))
            x.append(encoded_inputs)
            y.append(targets.to(self.device))

            y_hat = model.predict(inputs.to(self.device))
            x_samples = torch.repeat_interleave(
                encoded_inputs,
                repeats=self.settings.cce_settings.num_mc_samples,
                dim=0,
            )
            y_samples = model.sample(
                y_hat,
                num_samples=self.settings.cce_settings.num_mc_samples,
            ).flatten()
            x_prime.append(x_samples)
            y_prime.append(y_samples)

        x = torch.cat(x, dim=0)
        y = torch.cat(y).float()
        x_prime = torch.cat(x_prime, dim=0)
        y_prime = torch.cat(y_prime).float()

        return x, y, x_prime, y_prime

    def _get_kernel_functions(self, y: torch.Tensor) -> tuple[KernelFunction, KernelFunction]:
        if self.settings.cce_settings.input_kernel == "polynomial":
            x_kernel = polynomial_kernel
        else:
            x_kernel = self.settings.cce_settings.input_kernel

        if self.settings.cce_settings.output_kernel == "rbf":
            y_kernel = partial(rbf_kernel, gamma=(1 / (2 * y.float().var())).item())
        elif self.settings.cce_settings.output_kernel == "laplacian":
            y_kernel = partial(laplacian_kernel, gamma=(1 / (2 * y.float().var())).item())

        return x_kernel, y_kernel

    def _encode_tabular(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(self.image_preprocess(x), normalize=True)

    def _encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_text(x, normalize=True)

    def _load_or_compute_grid(self, grid_loader: DataLoader) -> torch.Tensor:
        if self._grid_cache_path.exists():
            return torch.load(self._grid_cache_path)

        if self.settings.dataset_type == DatasetType.TABULAR:
            encoder = self._encode_tabular
        elif self.settings.dataset_type == DatasetType.IMAGE:
            encoder = self._encode_image
        else:
            encoder = self._encode_text

        grid = torch.cat(
            [
                encoder(inputs.to(self.device))
                for inputs, _ in tqdm(grid_loader, desc="Encoding grid points", leave=False)
            ],
            dim=0,
        )
        torch.save(grid, self._grid_cache_path)
        return grid

    def _cleanup(self):
        if self._grid_cache_path.exists():
            self._grid_cache_path.unlink()

    @property
    def clip_model(self) -> CLIP:
        if self._clip_model is None:
            self._clip_model = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                device=self.device,
            )[0]
        return self._clip_model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return self._tokenizer
