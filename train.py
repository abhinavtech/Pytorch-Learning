"""
PyTorch Learning — Supervised Learning Examples

Usage:
    python train.py --model ffn   # Feedforward on MNIST
    python train.py --model cnn   # CNN on FashionMNIST
    python train.py --model rnn   # LSTM on sine wave

Dashboard runs at http://localhost:5000 during training.
"""

import argparse
import threading

import torch.nn as nn
import torch.optim as optim

from config import CNNConfig, FeedForwardConfig, RNNConfig
from data import get_fashion_mnist_loaders, get_mnist_loaders, get_sine_loaders
from engine import evaluate, train_one_epoch
from models import ConvNet, FeedForwardNet, LSTMPredictor
from utils import get_device, save_checkpoint, set_seed, setup_logger
from web.app import create_app, training_state

logger = setup_logger()

MODEL_DISPLAY_NAMES = {
    "ffn": "FeedForwardNet (MNIST)",
    "cnn": "ConvNet (FashionMNIST)",
    "rnn": "LSTMPredictor (Sine Wave)",
}


def _run_training_loop(model, train_loader, test_loader, criterion, optimizer,
                       device, total_epochs, checkpoint_path):
    """Shared training loop that updates the web dashboard."""
    for epoch in range(1, total_epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = evaluate(model, test_loader, criterion, device)

        training_state.update(
            current_epoch=epoch,
            train_loss=loss,
            test_loss=metrics["loss"],
            test_accuracy=metrics.get("accuracy"),
        )
        training_state.add_epoch({
            "epoch": epoch,
            "train_loss": loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics.get("accuracy"),
        })

        if metrics.get("accuracy") is not None:
            logger.info(
                f"Epoch {epoch} | Test Loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch} | Test MSE: {metrics['loss']:.6f}")

    save_checkpoint(model, optimizer, total_epochs, loss, checkpoint_path)
    training_state.update(status="completed")


def train_feedforward() -> None:
    device = get_device()
    cfg = FeedForwardConfig()
    training_state.update(
        model_name=MODEL_DISPLAY_NAMES["ffn"], status="training",
        total_epochs=cfg.epochs, current_epoch=0,
    )
    logger.info(f"Training FeedForwardNet on MNIST | device={device}")

    train_loader, test_loader = get_mnist_loaders(cfg.batch_size)
    model = FeedForwardNet(
        cfg.input_size, cfg.hidden1, cfg.hidden2, cfg.num_classes, cfg.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    _run_training_loop(
        model, train_loader, test_loader, criterion, optimizer,
        device, cfg.epochs, "checkpoints/ffn_mnist.pt",
    )


def train_cnn() -> None:
    device = get_device()
    cfg = CNNConfig()
    training_state.update(
        model_name=MODEL_DISPLAY_NAMES["cnn"], status="training",
        total_epochs=cfg.epochs, current_epoch=0,
    )
    logger.info(f"Training ConvNet on FashionMNIST | device={device}")

    train_loader, test_loader = get_fashion_mnist_loaders(cfg.batch_size)
    model = ConvNet(cfg.num_classes, cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    _run_training_loop(
        model, train_loader, test_loader, criterion, optimizer,
        device, cfg.epochs, "checkpoints/cnn_fashion.pt",
    )


def train_rnn() -> None:
    device = get_device()
    cfg = RNNConfig()
    training_state.update(
        model_name=MODEL_DISPLAY_NAMES["rnn"], status="training",
        total_epochs=cfg.epochs, current_epoch=0,
    )
    logger.info(f"Training LSTMPredictor on sine wave | device={device}")

    train_loader, test_loader = get_sine_loaders(cfg.seq_length, cfg.batch_size)
    model = LSTMPredictor(cfg.input_size, cfg.hidden_size, cfg.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    _run_training_loop(
        model, train_loader, test_loader, criterion, optimizer,
        device, cfg.epochs, "checkpoints/rnn_sine.pt",
    )


MODELS = {
    "ffn": train_feedforward,
    "cnn": train_cnn,
    "rnn": train_rnn,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Learning")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="Model to train: ffn (feedforward), cnn (convolutional), rnn (LSTM)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    args = parser.parse_args()

    set_seed(args.seed)

    # Start Flask dashboard in background thread
    app = create_app()
    server = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=args.port, debug=False),
        daemon=True,
    )
    server.start()
    logger.info(f"Dashboard running at http://localhost:{args.port}")

    # Run training
    training_state.reset()
    MODELS[args.model]()
    logger.info("Training complete. Dashboard still available — press Ctrl+C to exit.")

    # Keep process alive so dashboard remains accessible
    try:
        server.join()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
