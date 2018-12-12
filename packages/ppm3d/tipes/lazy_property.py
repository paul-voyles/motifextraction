import weakref
import functools


class LazyProperty(object):
    def __init__(self, method):
        self.method = method
        self.values = weakref.WeakKeyDictionary()

    def __set__(self, key, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, key):
        raise AttributeError("can't delete attribute")

    def __get__(self, key):
        try:
            return self.values[key]
        except KeyError:
            self.values[key] = self.method(key)
            return self.values[key]


def lazyproperty(method):
    """
    Args:
        method (function): the method to turn into a lazy property
    """
    prop = LazyProperty(method)

    @property
    @functools.wraps(method)
    def lazy(self):
        return prop.__get__(self)

    return lazy
