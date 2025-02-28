import logging
import os

def getQuickLogger(name: str, debug_folder: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(debug_folder, f"{name}.log"))
    logger.addHandler(fh)
    return logger

def clearLoggers():
    # Get the root logger
    root_logger = logging.getLogger()

    # Loop through all handlers and close them
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Optionally, you can also disable logging if needed
    logging.shutdown()