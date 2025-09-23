import random
from sycamore.data import MetadataDocument
from sycamore.utils.thread_local import ThreadLocalAccess, ThreadLocal, ADD_METADATA_TO_OUTPUT
from sycamore.utils.threading import run_coros_threadsafe


def test_run_coros():
    async def sometimes_recurse(n: int, tries: int = 0) -> int:
        if random.random() < 0.5:
            ThreadLocalAccess(ADD_METADATA_TO_OUTPUT).get().append(MetadataDocument(tries=tries))
            return n
        else:
            return await sometimes_recurse(n, tries + 1)

    nums = list(range(10))
    coros = [sometimes_recurse(n) for n in nums]
    meta = []
    with ThreadLocal(ADD_METADATA_TO_OUTPUT, meta):
        results = run_coros_threadsafe(coros)

    assert results == nums
    assert len(meta) == 10
