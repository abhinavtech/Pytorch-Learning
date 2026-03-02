from dataclasses import dataclass


@dataclass
class FeedForwardConfig:
    input_size: int = 784  # 28 * 28
    hidden1: int = 256
    hidden2: int = 128
    num_classes: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    dropout: float = 0.2


@dataclass
class CNNConfig:
    num_classes: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    dropout: float = 0.25


@dataclass
class RNNConfig:
    input_size: int = 1  # univariate time series
    hidden_size: int = 64
    num_layers: int = 2
    seq_length: int = 50  # look-back window
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 50
