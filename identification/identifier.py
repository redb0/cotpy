class Identifier:
    def __init__(self):
        self._model = None

        self._memory = 0
        self._sigma = 0

    def __repr__(self):
        pass

    def __str__(self):
        pass

    @property
    def model(self):
        return self._model

    @property
    def memory(self):
        return self._memory

    @property
    def sigma(self):
        return self._sigma
