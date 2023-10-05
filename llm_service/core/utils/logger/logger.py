import logging
import logging.config
import resource
import socket

from llm_service.core.utils.logger.config import LOGGING_CONFIG


class Logger:
    """
    Examples
    --------
    >>> from llm_service.core.utils.logger.logger import Logger
    >>> Logger(True).warning("test")
    C02G725ZMD6P [2022-07-07 16:07:27,522] {logger.py:32, warning} [10252.0M] WARNING - test
    >>> Logger(False).debug("test")
    >>> Logger(True).debug("test")
    C02G725ZMD6P [2022-07-07 16:12:34,804] {logger.py:44, debug} [10380.0M] DEBUG - ccc
    """

    def __init__(self, verbose: bool = False) -> None:
        logging.config.dictConfig(LOGGING_CONFIG)
        logger = logging.getLogger("dev") if verbose else logging.getLogger("prod")

        self.logger = logging.LoggerAdapter(
            logger,
            {
                "hostname": socket.gethostname(),
                "memory_usage": "{:.01f}M".format(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                ),
            },
        )

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)
