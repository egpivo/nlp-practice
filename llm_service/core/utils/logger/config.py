# Custom logging configure
# Ref: https://stackoverflow.com/a/7507842
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] {%(filename)s:%(lineno)d, %(funcName)s} [%(memory_usage)s] %(levelname)s -  %(message)s"
        },
        "detailed": {
            "format": "%(hostname)s [%(asctime)s] {%(filename)s:%(lineno)d, %(funcName)s} [%(memory_usage)s] %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "root": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        "dev": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        "prod": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
