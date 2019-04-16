from typing import Any
import numpy as np
from torch.utils.data import Dataset


class AlbuDataset(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            transforms,
            key: str = "image"
    ):
        self.dataset = dataset
        self.transforms = transforms
        self.key = key

    def __getitem__(self, index: int) -> Any:
        item, label = self.dataset[index]
        image = np.array(item)
        param = {self.key: image}
        result = self.transforms(**param)[self.key]
        return result, label

    def __len__(self):
        return self.dataset.__len__()
