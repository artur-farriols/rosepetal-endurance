import sys
import functools

from loguru import logger

logger_initialized = {}


@functools.lru_cache()
def get_logger(name='ROSEPETAL SERVER', log_file=None):
    """Initialize and get a logger by name using loguru.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str, optional): Path to log file. If specified, logs will also be written to this file.
    Returns:
        loguru.logger: The expected logger.
    """
    global logger_initialized

    if name not in logger_initialized:
        # Setup loguru logger
        logger.remove()  # Remove default handler
        logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}", level="INFO")
        
        if log_file:
            logger.add(log_file, rotation="10 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}", level="INFO")

        logger_initialized[name] = True
    
    # Note: With loguru, there's no need to return a specific logger instance based on the name since it manages that internally.
    # However, to maintain the interface, we could bind the logger with the provided name using `.bind()`.
    return logger.bind(name=name)