import os
import sys
import logging
import logging.handlers as handlers

import a_Common.log_colorer as log_colorer

DEFAULT_LOG_FORMAT = logging.Formatter(
    '[%(thread)d] %(filename)s:%(lineno)d : %(levelname)5s\n'
    '[%(thread)d] %(asctime)s |  %(name)s.%(funcName)s:%(message)s'
)
CLIENT_LOG_FORMAT = logging.Formatter('%(asctime)s.%(msecs)03d | %(filename)20s:%(lineno)4d | %(levelname)5s '
                                      '%(funcName)s: %(message)s', "%H:%M:%S"
                                      )

LOG_FOLDER_NAME = '_logs'
LOG_FOLDER_PATH = os.path.join(os.getcwd(), LOG_FOLDER_NAME)
os.makedirs(LOG_FOLDER_PATH, exist_ok=True)

FILE_LOGGING_LEVEL = logging.DEBUG
CLIENT_CONSOLE_LEVEL = logging.DEBUG
SERVER_CONSOLE_LEVEL = logging.INFO
CLIENT_DB_LOGGING_LEVEL = logging.INFO
SERVER_DB_LOGGING_LEVEL = logging.DEBUG

CONSOLE_FORMAT = logging.Formatter('%(asctime)s | %(message)s')
CONSOLE_HANDLER = log_colorer.ColoredConsoleHandler(sys.stdout)
CONSOLE_HANDLER.setFormatter(CONSOLE_FORMAT)
CONSOLE_HANDLER.setLevel(CLIENT_CONSOLE_LEVEL)


def get_new_logger(leaf, file_name="KaryPy", file_max_mb=64, file_count=16):
    file_path = os.path.join(LOG_FOLDER_PATH, '%s.log' % file_name)
    file_handle = handlers.RotatingFileHandler(file_path, maxBytes=1024 * 1024 * file_max_mb, backupCount=file_count)
    file_handle.setFormatter(fmt=DEFAULT_LOG_FORMAT)
    file_handle.setLevel(FILE_LOGGING_LEVEL)
    logger = logging.getLogger(leaf)
    logger.addHandler(file_handle)
    logger.setLevel(FILE_LOGGING_LEVEL)
    logger.addHandler(CONSOLE_HANDLER)
    logger.debug('\t\t\t\t\t[%s LOGGER START] = %s\n' % (leaf, file_path))
    return logger


DEFAULT_LOGGER_NAME = "mainlogger"

LOGGER = get_new_logger(leaf=DEFAULT_LOGGER_NAME, file_name='common')
