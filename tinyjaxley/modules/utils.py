def recurse(f):
    def wrapper(self, *args, **kwargs):
        out = [recurse(f)(sm, *args, **kwargs) for sm in self]
        if self.submodules is None:
            return f(self, *args, **kwargs)
        if out[0] == None:
            return None
        return out
    return wrapper