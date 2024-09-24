from typing import TYPE_CHECKING
from mypy import api
from sycamore.utils.import_utils import requires_modules


# The actual library doesn't matter here.
@requires_modules("apted", extra="eval")
def require_fn() -> int:
    return 42


res = require_fn()
if TYPE_CHECKING:
    reveal_type(res)  # noqa: F821


# This test fails prior to adding generic (ParamSpec and TypeVar) type annotations to the
# requires_modules decorator, as the revealed type is "Any".
def test_mypy_type():
    mypy_res = api.run([__file__])
    assert 'Revealed type is "builtins.int"' in mypy_res[0]
