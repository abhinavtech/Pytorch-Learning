"""Flask web dashboard for monitoring training progress."""

import threading
from dataclasses import dataclass, field

from flask import Flask, jsonify, render_template


@dataclass
class TrainingState:
    """Thread-safe container for training metrics."""

    model_name: str = ""
    status: str = "idle"  # idle | training | completed
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    test_loss: float = 0.0
    test_accuracy: float | None = None
    history: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k != "_lock":
                    setattr(self, k, v)

    def add_epoch(self, epoch_data: dict) -> None:
        with self._lock:
            self.history.append(epoch_data)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model_name": self.model_name,
                "status": self.status,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "train_loss": self.train_loss,
                "test_loss": self.test_loss,
                "test_accuracy": self.test_accuracy,
                "history": list(self.history),
            }

    def reset(self) -> None:
        with self._lock:
            self.model_name = ""
            self.status = "idle"
            self.current_epoch = 0
            self.total_epochs = 0
            self.train_loss = 0.0
            self.test_loss = 0.0
            self.test_accuracy = None
            self.history = []


# Global state shared between training thread and Flask
training_state = TrainingState()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/status")
    def status():
        return jsonify(training_state.snapshot())

    return app
