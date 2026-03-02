# PyTorch Learning

Production-ready PyTorch project demonstrating supervised learning fundamentals:
**feedforward networks**, **CNNs**, and **RNNs with backpropagation through time**.

## Models

| Model | Architecture | Dataset | Task |
|-------|-------------|---------|------|
| `ffn` | FeedForwardNet (Linear + ReLU + Dropout) | MNIST | Digit classification |
| `cnn` | ConvNet (Conv2d + MaxPool) | FashionMNIST | Image classification |
| `rnn` | LSTMPredictor (2-layer LSTM) | Synthetic sine wave | Next-value prediction |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py --model ffn   # Feedforward on MNIST (~97% accuracy)
python train.py --model cnn   # CNN on FashionMNIST (~90% accuracy)
python train.py --model rnn   # LSTM on sine wave (MSE < 0.001)
```

Options:
- `--seed N` — Set random seed (default: 42)

## Project Structure

```
├── train.py              # Unified entry point
├── config.py             # Hyperparameters (dataclasses)
├── models/               # Model definitions (nn.Module)
├── data/                 # Dataset loaders
├── engine/               # Training loop & evaluation
└── utils/                # Seed, device, checkpoints, logging
```

## Key Concepts Demonstrated

- **Backpropagation**: `loss.backward()` computes gradients via autograd
- **BPTT**: LSTM unrolls across time steps; gradients flow back through the sequence
- **Train/eval modes**: `model.train()` enables dropout; `model.eval()` disables it
- **Device handling**: Automatic CUDA/MPS/CPU selection
- **Reproducibility**: Seeded random states across Python, NumPy, and PyTorch
- **Checkpointing**: Save and resume training with `state_dict`
