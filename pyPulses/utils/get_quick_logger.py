import logging
import os

def getQuickLogger(name: str, debug_folder: str) -> logging.Logger:
    """
    Lazy method to get a logger for and instrument

    Parameters
    ----------
    name : str
        logger name
    debug_folder : str
        path to the folder for the log file
    
    Returns
    -------
    Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(debug_folder, f"{name}.log"))
    logger.addHandler(fh)
    return logger

# NOT SURE IF THIS WORKS PROPERLY
def clearLoggers():
    """Attempt to get rid of all loggers."""
    # Get the root logger
    root_logger = logging.getLogger()

    # Loop through all handlers and close them
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    logging.shutdown()
