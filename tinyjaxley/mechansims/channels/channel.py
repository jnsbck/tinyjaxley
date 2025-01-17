class Channel:
    def __init__(self, params = {}, states = {}):
        self.states = states
        self.params = params

    @property 
    def name(self): return self.__class__.__name__.lower()

    def i(self, u, p, t): return 0
    def du(self, u, p, t): return {}
    def init(self, u, p): return {}
