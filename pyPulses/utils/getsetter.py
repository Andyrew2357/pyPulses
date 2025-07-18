"""
Package a getter and a setter into one function (Acts as a getter when no
parameter is passed.)
"""

from typing import Any, Callable

def getSetter(get: Callable[[], Any], 
              set: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Package a getter and setter into one function. The resulting function acts
    as a getter when no parameter is passed and a setter otherwise.

    Parameters
    ----------
    get, set : Callable
        getter and setter functions

    Returns
    -------
    Callable
    """
    def P(x: Any = None):
        if x is None:
            return get()
        set(x)

    return P
