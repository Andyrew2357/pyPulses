from typing import Any, Callable

def sample(f: Callable[[], float], N: int):
    """Sample the same value N times, returning the average"""
    def f_sampled():    
        s = 0
        for _ in range(N):
            s += f()
        return s/N
    
    return f_sampled

def log_return(f: Callable, logger):
    """Log the return values of a function"""
    def f_logging_return(*args, **kwargs):
        res = f(*args, **kwargs)
        logger.debug(f"Function {f.__name__} returned: {res}")
        return res
    
    return f_logging_return

def log_args(f:Callable, logger):
    """Log the arguments passed to a function"""
    def f_logging_args(*args, **kwargs):
        logger.debug(
            f"Function {f.__name__} received: args = {args}, kwargs = {kwargs}"
        )
        return f(*args, **kwargs)
    
    return f_logging_args
