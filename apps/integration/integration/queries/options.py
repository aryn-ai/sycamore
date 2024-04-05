from collections.abc import Iterable


class Option:
    def __init__(self, name, values):
        if isinstance(values, str) or not isinstance(values, Iterable):
            values = [values]
        if not isinstance(values, list):
            values = list(values)
        self._name = name
        self._values = values
        self._current_pos = 0

    def get(self):
        return self._values[self._current_pos]

    def get_name(self):
        return self._name

    def advance(self):
        self._current_pos += 1
        if self._current_pos == len(self._values):
            self._current_pos = 0
            return True
        return False


class BooleanOption(Option):
    def __init__(self, name):
        super().__init__(name, [False, True])


class OptionSet(Iterable):

    def __iter__(self):
        finished = False
        while not finished:
            yield {op.get_name(): op.get() for op in self._options}
            i = 0
            op = self._options[i]
            while op.advance():
                i += 1
                if i == len(self._options):
                    finished = True
                    break
                op = self._options[i]

    def __init__(self, *options):
        self._options = options
        assert len(options) == len({op.get_name() for op in options}), "Options must have unique names"
