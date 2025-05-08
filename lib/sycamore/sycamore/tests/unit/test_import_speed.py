import subprocess
import datetime


def test_sycamore_import_speed():
    # Attention future developer: You're here, your PR just made this test fail, you're thinking
    # "I'll just increase this by a little bit, it's not a big deal."  You're probably not even
    # responsible for most of the slowdown. Please, take the time to fix it and not let it get
    # "just a little bit worse."  That's the way that we get back up to taking >1s to import a
    # library when it should take <0.1s.
    #
    # There are several techniques for improving import speed:
    # 1. Making dependencies lazy -- move the import statement into the class init or
    #    function and the dependency won't be loaded unless the function is called.
    # 2. Removing typing only dependencies -- add the imports under an if TYPE_CHECKING block,
    #    and add "" around the conditionally imported types.
    # 3. Removing things from __init__.py -- chhange the user inputs to import from the underlying
    #    file, and remove them from __init__.py so they aren't unconditionally imported.
    # 4. Split out partial dependencies, e.g. sycamore/llms/config.py which just holds some
    #    configuration information needed by callers to determine valid LLMs, instead of importing
    #    all of the LLM logic.

    # elapsed time is charging for starting a python process also; just the import time is
    # shorter. Current constant was emperically determined after optimization. The max is 1.5x the
    # slowest run seen in github after setting max_elapsed = 0.001. If we fix the remaining LLM
    # import slowness the constant can be reduced.

    max_elapsed = 0.364227 * 1.5

    all_elapsed = []
    # First run is sometimes slow, give it 10 tries to get under the target
    for i in range(10):
        start = datetime.datetime.now()
        ret = subprocess.run(["python", "-c", "import sycamore"])
        assert ret.returncode == 0
        elapsed = (datetime.datetime.now() - start).total_seconds()
        if elapsed <= max_elapsed:
            break

        all_elapsed.append(elapsed)

    if elapsed > max_elapsed:
        subprocess.run(["python", "-c", "import import_timer; import sycamore; import_timer.show()"])
        print(f"ERROR: All runtimes > {max_elapsed}: {all_elapsed}")
        raise ValueError("Import too slow")
