from pathlib import Path

from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import AAFDataset
from probcal.data_modules.probcal_datamodule import ProbcalDataModule


class AAFDataModule(ProbcalDataModule):

    IMG_SIZE = 224

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        self.surface_image_path = surface_image_path

    def prepare_data(self) -> None:
        # download and extract the data
        AAFDataset(self.root_dir, split="train")

    def setup(self, stage):
        if self.al_setup:
            return
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )  # Check the mean and variance of our images
        to_tensor = ToTensor()
        train_transforms = Compose([resize, augment, to_tensor, normalize])
        inference_transforms = Compose([resize, to_tensor, normalize])

        self.train = AAFDataset(
            self.root_dir,
            split="train",
            transform=train_transforms,
            surface_image_path=self.surface_image_path,
        )
        self.val = AAFDataset(
            self.root_dir,
            split="val",
            transform=inference_transforms,
            surface_image_path=self.surface_image_path,
        )
        self.test = AAFDataset(
            self.root_dir,
            split="test",
            transform=inference_transforms,
            surface_image_path=self.surface_image_path,
        )
