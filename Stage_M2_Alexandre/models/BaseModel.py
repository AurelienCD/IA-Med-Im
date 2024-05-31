"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, task_name: str, dataset_name: str) -> None:
        super().__init__()
        self.task_name = task_name
        self.dataset_name = dataset_name

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError

    def save(self) -> None:
        filename = f'./weights/{self.__class__.__name__}_{self.task_name}_{self.dataset_name}.pt'
        torch.save(self.state_dict(), filename)

    def load_weights(self, file_path) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(file_path, map_location=device))
