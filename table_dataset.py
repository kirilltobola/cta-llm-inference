import pandas as pd

from torch.utils.data import Dataset
from datasets import load_dataset

class TableDataset(Dataset):
    def __init__(self, path, data_column="column_data", label_column="label"):
        self.dataset = pd.DataFrame(load_dataset(path, split="test"))
        self.data_column = data_column
        self.label_column = label_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "data": self.dataset.iloc[idx][self.data_column],
            "label": self.dataset.iloc[idx][self.label_column]   
        }
