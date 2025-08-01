from pathlib import Path
from typing import Callable
from typing import Literal

from probcal.custom_datasets.mixins import DatasetIndexMixin

import pandas as pd
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset



class EVADataset(DatasetIndexMixin, Dataset):
    """
    EVA dataset with images voted on how asthetic they are (labeled with the average asthetic score for each image).
    The original dataset is available at https://github.com/kang-gnak/eva-dataset.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ) -> None:
        """Create an instance of the EVA dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and count in __getitem__.
        """
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.surface_image_path = surface_image_path
        self.split = split

        self.root_dir = Path(root_dir)
        self.labels_csv = self.root_dir.joinpath("labels.csv")
        self.image_dir = self.root_dir.joinpath("images")

        if not self._check_for_eva_data():
            raise FileNotFoundError(
                "Dataset is not present in the specified location. Contact the authors for access."
            )

        self.labels_df = pd.read_csv(self.labels_csv)
        self.instances = self._get_instances_df()

    def _get_instances_df(self) -> pd.DataFrame:
        mask = self.labels_df["split"] == self.split
        instances = {"image_path": [], "avg_score": []}
        for _, row in self.labels_df[mask].iterrows():
            image_path = str(self.image_dir / f"{row['image_id']}.jpg")
            score = float(row["avg_score"])
            instances["image_path"].append(image_path)
            instances["avg_score"].append(score)
        return pd.DataFrame(instances)

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        try:
            row = self.instances.iloc[idx]
        except TypeError:
            row = self.instances.iloc[idx.item()]
        image_path = row["image_path"]
        image = Image.open(image_path)
        image = self._ensure_rgb(image)
        score = row["avg_score"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            score = self.target_transform(score)
        if self.surface_image_path:
            return image, (image_path, score)
        else:
            return image, score

    def __len__(self):
        return len(self.instances)

    def _ensure_rgb(self, image: PILImage):
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _check_for_eva_data(self) -> bool:
        return self.root_dir.exists() and self.image_dir.exists() and self.labels_csv.exists()
