from pathlib import Path
from typing import Union
from PIL import Image

from torch.utils.data import Dataset


class ImageFolderWithoutSubfolders(Dataset):
    """ImageFolder dataset without subfolders

    This type of dataset is useful for unsupervised image classification tasks etc"""

    def __init__(self, folder: Union[str, Path], transform=None):
        """

        Args:
            folder: path to folder

            transform: transform to apply"""

        folder = Path(folder)
        self._transform = transform
        self._image_paths = [x for x in folder.glob('*') if x.is_file()]

    def __getitem__(self, index: int):
        image_path = self._image_paths[index]
        x = Image.open(image_path).convert('RGB')

        if self._transform is not None:
            x = self._transform(x)
        return x, -1

    def __len__(self) -> int:
        return len(self._image_paths)
