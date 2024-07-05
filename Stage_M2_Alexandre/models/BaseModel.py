"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError

    def save(self, filename: str) -> None:
        filename = f'./weights/{self.__class__.__name__}_{filename}.pt'
        torch.save(self.state_dict(), filename)

    def load_weights(self, file_path) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.load_state_dict(torch.load(file_path, map_location=device))
