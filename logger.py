# python: >=3.5
# encoding: utf-8

import logging
import sys


class Logger(object):
    """Logger for any program."""

    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, logger_file, log_level='info', filemode='a'):
        if log_level == "debug":
            logging_level = logging.DEBUG
        elif log_level == "info":
            logging_level = logging.INFO
        elif log_level == "warn":
            logging_level = logging.WARN
        elif log_level == "error":
            logging_level = logging.ERROR
        else:
            raise TypeError(
                "No logging type named %s, candidate is: "
                "info, warn, debug, error")
        logging.basicConfig(
            filename=logger_file, level=logging_level,
            format='%(asctime)s : %(levelname)s  %(message)s',
            filemode=filemode, datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def debug(msg):
        """Log debug message
            msg: Message to log
        """
        logging.debug(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def info(msg):
        """"Log info message
            msg: Message to log
        """
        logging.info(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def warn(msg):
        """Log warn message
            msg: Message to log
        """
        logging.warning(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def error(msg):
        """Log error message
            msg: Message to log
        """
        logging.error(msg)
        sys.stderr.write(msg + "\n")


if __name__ == '__main__':
    logger_file = './test.log'
    log_level = 'info'

    logger = Logger(logger_file, log_level, filemode='w')
    logger.info('hello world')
