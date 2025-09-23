import asyncio
import threading
from sycamore.utils.thread_local import ThreadLocal, ThreadLocalAccess, ADD_METADATA_TO_OUTPUT


def _run_new_thread(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def run_coros_threadsafe(coros):
    new_loop = asyncio.new_event_loop()
    t = threading.Thread(target=_run_new_thread, args=(new_loop,), daemon=True)
    t.start()

    metadata = []

    async def _gather_coros(coros):
        # Exfiltrate metadata documents from inner thread
        with ThreadLocal(ADD_METADATA_TO_OUTPUT, metadata):
            tasks = [new_loop.create_task(c) for c in coros]
            return await asyncio.gather(*tasks)

    fut = asyncio.run_coroutine_threadsafe(_gather_coros(coros), loop=new_loop)
    results = fut.result()
    new_loop.call_soon_threadsafe(new_loop.stop)
    t.join()
    new_loop.close()
    tls = ThreadLocalAccess(ADD_METADATA_TO_OUTPUT)
    if tls.present():
        tls.get().extend(metadata)
    return results
