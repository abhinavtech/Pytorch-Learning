import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SineWaveDataset(Dataset):
    """Synthetic sine wave dataset for time series prediction."""

    def __init__(self, num_samples: int = 1000, seq_length: int = 50) -> None:
        self.seq_length = seq_length
        t = np.linspace(0, num_samples * 0.1, num_samples + seq_length)
        self.data = np.sin(t).astype(np.float32)

    def __len__(self) -> int:
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx : idx + self.seq_length]).unsqueeze(-1)
        y = torch.tensor(self.data[idx + self.seq_length]).unsqueeze(-1)
        return x, y


def get_sine_loaders(
    seq_length: int = 50, batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    """Return train and test DataLoaders for sine wave prediction."""
    train_dataset = SineWaveDataset(num_samples=3000, seq_length=seq_length)
    test_dataset = SineWaveDataset(num_samples=500, seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
