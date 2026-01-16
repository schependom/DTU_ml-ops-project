"""PyTorch Lightning logger adapter for Loguru integration.

This module provides a custom Lightning Logger that routes all training
logs through Loguru, enabling consistent logging format across the
entire application (both Lightning internals and custom code).

Usage:
    from loguru import logger
    from ml_ops_project.pl_logging import LoguruLightningLogger

    loguru_adapter = LoguruLightningLogger(logger)
    trainer = pl.Trainer(logger=[loguru_adapter, wandb_logger])
"""

from typing import Any

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


class LoguruLightningLogger(Logger):
    """PyTorch Lightning Logger that delegates to a Loguru logger instance.

    This adapter allows Lightning's built-in logging (hyperparameters, metrics,
    training status) to flow through Loguru's formatting and file handlers.

    Args:
        logger_instance: A configured Loguru logger (typically the global `logger`).
        save_dir: Base directory for logs (used by Lightning's directory structure).
        version: Optional version string for experiment tracking.

    Attributes:
        name: Logger name, always "loguru".
        version: Experiment version string.
        root_dir: Computed log directory path (required by Lightning interface).
    """

    def __init__(
        self,
        logger_instance: Any,
        save_dir: str = "logs",
        version: str | None = None,
    ) -> None:
        super().__init__()
        self._logger = logger_instance
        self._save_dir = save_dir
        self._version = version
        self._name = "loguru"

    @property
    def name(self) -> str:
        """Return the logger name."""
        return self._name

    @property
    def version(self) -> str:
        """Return the experiment version."""
        return self._version if self._version else "0"

    @property
    def root_dir(self) -> str:
        """Return the root directory for this logger.

        Required by Lightning interface, but Loguru manages its own file paths
        configured elsewhere (in train.py via logger.add()).
        """
        return f"{self._save_dir}/{self.name}/{self.version}"

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters at the start of training.

        Decorated with @rank_zero_only to prevent duplicate logs in
        distributed training (only rank 0 process logs).

        Args:
            params: Dictionary of hyperparameter names and values.
        """
        self._logger.info(f"Hyperparameters: {params}")

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics during training/validation/test.

        Args:
            metrics: Dictionary of metric names and values (e.g., {"loss": 0.5}).
            step: Current training step or epoch number.
        """
        self._logger.info(f"Step {step}: {metrics}")

    def save(self) -> None:
        """Save logger state (no-op for Loguru, which handles its own flushing)."""
        pass

    def finalize(self, status: str) -> None:
        """Called at the end of training.

        Args:
            status: Final training status (e.g., "success", "failed", "interrupted").
        """
        self._logger.info(f"Training finished with status: {status}")
