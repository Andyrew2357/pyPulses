from typing import Any, Callable
from logging import Logger
import time
import functools

def compose_decorators(*decorators: Callable) -> Callable:
    """
    Compose multiple decorators into a single decorator, preserving metadata.

    Parameters
    ----------
    *decorators : Callable
        Any number of decorators to apply. The first one listed is applied last.

    Returns
    -------
    Callable
        A new decorator that applies all of the given decorators in order.
    """

    def composed(f: Callable) -> Callable:
        original = f
        for decorator in reversed(decorators):
            f = decorator(f)
        return functools.wraps(original)(f)

    return composed

def sample(N: int, wait: float = 0) -> Callable:
    """
    Decorator that modifies a getter to sample a value several times and return 
    the mean.

    Parameters
    ----------
    N : int
        number of samples to take.
    wait : float, default=0.0
        seconds between taking samples
    """
    
    def wrap(f: Callable) -> Callable:
        @functools.wrap(f)
        def f_sampled(*args, **kwargs):    
            s = 0
            for _ in range(N):
                s += f(*args, **kwargs)
                time.sleep(wait)
            return s/N
        
        return f_sampled
    
    return wrap

def log_return(logger: Logger) -> Callable:
    """
    Decorator that modifies a function to log the return values.
    
    Parameters
    ----------
    logger : Logger
    """
    
    def wrap(f: Callable) -> Callable:
        @functools.wrap(f)
        def f_logging_return(*args, **kwargs):
            res = f(*args, **kwargs)
            logger.debug(f"Function {f.__name__} returned: {res}")
            return res
        
        return f_logging_return
    
    return wrap

def log_args(logger: Logger) -> Callable:
    """
    Decorator that modifies a function to log the arguments passed.
    
    Parameters
    ----------
    logger : Logger
    """
    
    def wrap(f: Callable) -> Callable:
        @functools.wrap(f)
        def f_logging_args(*args, **kwargs):
            logger.debug(
                f"Function {f.__name__} received: args = {args}, kwargs = {kwargs}"
            )
            return f(*args, **kwargs)   
         
        return f_logging_args
    
    return wrap

def limit_setter(min: float, max:float) -> Callable:
    """
    Decorator that limits the range of a function that takes a float as its 
    argument.

    Parameters
    ----------
    min : float
    max : float
    """

    def wrap(f: Callable[[float], Any]) -> Callable:
        @functools.wrap(f)
        def f_limited(x: float):
            if x < min or x > max:
                raise ValueError(
                    f"{f.__name__} takes values between {min} and {max}."
                )
            
            return f(x)
        
        return f_limited
    
    return wrap
