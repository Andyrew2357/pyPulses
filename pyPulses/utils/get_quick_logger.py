import logging
import os

def getQuickLogger(name: str, debug_folder: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(debug_folder, f"{name}.log"))
    logger.addHandler(fh)
    return logger
