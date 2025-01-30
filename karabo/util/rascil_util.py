""" Offers functions to handle some of the RASCIL output. """
import logging
import os
from logging import LogRecord
from sysconfig import get_path

DATA_DIR_WARNING_MESSAGE = "The RASCIL data directory is not available - continuing but any simulations will fail"  # noqa: E501
DATA_DIR_WARNING_PATH_TO_MODULE = os.path.join(
    get_path("platlib"),
    "rascil",
    "processing_components",
    "util",
    "installation_checks.py",
)


def filter_data_dir_warning_message() -> None:
    """Avoid unnecessary RASCIL warning that confuses users.

    Currently the following RASCIL warning is suppressed:

    The RASCIL data directory is not available - continuing but any simulations will
    fail

    ...which pops up because we don't download the RASCIL data directory.
    To the best of our knowledge, we don't need the data directory (31.07.2024).
    We can therefore ignore this warning and avoid unnecessarily alerting users with it.
    """

    def filter_message(record: LogRecord) -> bool:
        if record.getMessage() == DATA_DIR_WARNING_MESSAGE:
            return False
        else:
            return True

    # Install filter on the RASCIL sub-module logger that logs the warning.
    # This logger is instantiated with __file__ as its name.
    logger_name = DATA_DIR_WARNING_PATH_TO_MODULE
    logger = logging.getLogger(logger_name)
    logger.addFilter(filter_message)
