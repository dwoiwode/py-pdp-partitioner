from functools import wraps
from unittest import SkipTest


def skip_abstract(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if "Abstract" in self.__class__.__name__:
            raise SkipTest("Ignore abstract method")
        return f(self, *args, **kwargs)

    return wrapper
