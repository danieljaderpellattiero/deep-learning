import logging

class CLogger:
		def __init__(self, log_file: str = "", level: int = logging.INFO, muted: bool = False):
				self.muted = muted
				self.log_file = log_file
				self._setup_logging(level)

		def _setup_logging(self, level: int) -> None:
				logging.basicConfig(
				filename=self.log_file,
				level=level,
				format='%(asctime)s - %(levelname)s - %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S'
				)
				self.logger = logging.getLogger(__name__)

		def info(self, message: str) -> None:
				if self.muted:
						return
				self.logger.info(message)

		def warning(self, message: str) -> None:
				if self.muted:
						return
				self.logger.warning(message)

		def error(self, message: str) -> None:
				if self.muted:
						return
				self.logger.error(message)
