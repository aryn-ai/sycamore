from functools import reduce
from typing import Union


def combine_strs_min_newline(*strs: Union[str, None]) -> str:
    def combine_str_min_newline(str1: str, str2: str) -> str:
        if str1.endswith("\n") or str2.startswith("\n"):
            return str1 + str2
        else:
            return str1 + "\n" + str2

    def safe_filter(strs):
        filtered = filter(None, strs)
        empty = True
        while True:
            try:
                n = next(filtered)
                empty = False
                yield n
            except StopIteration:
                break
        if empty:
            yield ""

    return reduce(combine_str_min_newline, safe_filter(strs))
