import logging
import logging.config
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "main_file": {
                "class": "logging.FileHandler",
                "filename": os.path.join(LOG_DIR, "main.log"),
                "formatter": "standard",
                "level": "DEBUG",
                "encoding": "utf8"
            },
            "tools_file": {
                "class": "logging.FileHandler",
                "filename": os.path.join(LOG_DIR, "tools.log"),
                "formatter": "standard",
                "level": "DEBUG",
                "encoding": "utf8"
            },
            "all_file": {
                "class": "logging.FileHandler",
                "filename": os.path.join(LOG_DIR, "all.log"),
                "formatter": "standard",
                "level": "DEBUG",
                "encoding": "utf8"
            },
            "agent_file": {
                "class": "logging.FileHandler",
                "filename": os.path.join(LOG_DIR, "agent.log"),
                "formatter": "standard",
                "level": "DEBUG",
                "encoding": "utf8"
            }
        },
        "loggers": {
            "main": {
                "handlers": ["main_file", "all_file"],
                "level": "DEBUG",
                "propagate": False
            },
            "tools": {
                "handlers": ["tools_file", "all_file"],
                "level": "DEBUG",
                "propagate": False
            },
            "agent": {
                "handlers": ["agent_file", "all_file"],
                "level": "DEBUG",
                "propagate": False
            }

        }
    }


def setup_logging():
    logging.config.dictConfig(get_logger_config())
