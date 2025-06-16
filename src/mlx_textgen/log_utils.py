import logging
import sys
import logging.handlers

def get_logger(name: str) -> logging.Logger:
    from .utils import get_package_cache_dir
    import os

    logging.basicConfig(format='[(%(levelname)s) %(asctime)s]: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_dir = os.path.join(get_package_cache_dir(), 'logs', 'api.log')
        if not os.path.exists(os.path.dirname(file_dir)):
            os.makedirs(os.path.dirname(file_dir))
        handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(get_package_cache_dir(), 'logs', 'api.log'),
            when='H',
            interval=1,
            backupCount=96
        )
        # Create a formatter to define the log entry format
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for standard output
        console_handler.setFormatter(formatter)

        # Add the custom handler to the logger
        logger.addHandler(handler)
        logger.addHandler(console_handler)
    return logger