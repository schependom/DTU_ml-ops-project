from typing import Any, Dict, Optional

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


class LoguruLightningLogger(Logger):
    def __init__(self, logger_instance, save_dir: str = "logs", version: Optional[str] = None):
        super().__init__()
        # Store the specific logger instance you configured
        self._logger = logger_instance
        self._save_dir = save_dir
        self._version = version
        self._name = "loguru"

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version if self._version else "0"

    @property
    def root_dir(self) -> str:
        # This property is required by PL, but Loguru ignores it
        # because you've already defined your file paths in the main script.
        return f"{self._save_dir}/{self.name}/{self.version}"

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]):
        # Uses your specific logger instance
        self._logger.info(f"Hyperparameters: {params}")

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        # Uses your specific logger instance
        self._logger.info(f"Step {step}: {metrics}")

    def save(self):
        pass

    def finalize(self, status: str):
        self._logger.info(f"Training finished with status: {status}")
