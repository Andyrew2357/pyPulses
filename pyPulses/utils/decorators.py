from typing import Callable
from logging import Logger
import time

def sample(N: int, wait: float = 0) -> Callable:
    """
    Sample a value N times, waiting 'wait' seconds each time, and return the 
    mean.
    """
    
    def wrap(f: Callable) -> Callable:

        def f_sampled(*args, **kwargs):    
            s = 0
            for _ in range(N):
                s += f(*args, **kwargs)
                time.sleep(wait)
            return s/N
        
        return f_sampled
    
    return wrap

def log_return(logger: Logger) -> Callable:
    """Log the return values of a function"""
    
    def wrap(f: Callable) -> Callable:

        def f_logging_return(*args, **kwargs):
            res = f(*args, **kwargs)
            logger.debug(f"Function {f.__name__} returned: {res}")
            return res
        
        return f_logging_return
    
    return wrap

def log_args(logger: Logger) -> Callable:
    """Log the arguments passed to a function"""
    
    def wrap(f: Callable) -> Callable:

        def f_logging_args(*args, **kwargs):
            logger.debug(
                f"Function {f.__name__} received: args = {args}, kwargs = {kwargs}"
            )
            return f(*args, **kwargs)   
         
        return f_logging_args
    
    return wrap
