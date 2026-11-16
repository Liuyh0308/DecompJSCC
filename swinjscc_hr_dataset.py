"""High-resolution image datasets used by DecompJSCC training and evaluation."""

from __future__ import annotations

import glob
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def split_path_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def collect_image_files(directories: list[str]) -> list[str]:
    files: list[str] = []
    for directory in directories:
        for pattern in IMAGE_PATTERNS:
            files.extend(glob.glob(str(Path(directory) / "**" / pattern), recursive=True))
    return sorted(set(files))


class HRRandomCropImageDataset(Dataset):
    """RGB training images sampled as random square crops."""

    def __init__(self, image_dirs: list[str], crop_size: int = 256):
        self.image_files = collect_image_files(image_dirs)
        if not self.image_files:
            raise FileNotFoundError(f"No HR training images found under: {image_dirs}")
        self.transform = transforms.Compose(
            [transforms.RandomCrop((int(crop_size), int(crop_size))), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image = Image.open(self.image_files[index]).convert("RGB")
        return self.transform(image), 0


class HRCroppedEvalImageDataset(Dataset):
    """RGB evaluation images center-cropped to a model-compatible multiple."""

    def __init__(self, image_dirs: list[str], crop_multiple: int = 128):
        self.image_files = collect_image_files(image_dirs)
        if not self.image_files:
            raise FileNotFoundError(f"No HR validation images found under: {image_dirs}")
        self.crop_multiple = int(crop_multiple)
        if self.crop_multiple <= 0:
            raise ValueError("crop_multiple must be positive")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        crop_width = width - width % self.crop_multiple
        crop_height = height - height % self.crop_multiple
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(
                f"Image {image_path} is smaller than crop multiple {self.crop_multiple}: {width}x{height}"
            )
        image = transforms.CenterCrop((crop_height, crop_width))(image)
        return transforms.ToTensor()(image), Path(image_path).name
