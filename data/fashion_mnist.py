from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fashion_mnist_loaders(
    batch_size: int = 64, data_dir: str = "data_cache"
) -> tuple[DataLoader, DataLoader]:
    """Return train and test DataLoaders for FashionMNIST."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_set = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
