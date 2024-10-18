#from typing_extensions import assert_type
#from sycamore.utils.import_utils import requires_modules
#
#
## The actual library doesn't matter here.
#@requires_modules("apted", extra="eval")
#def require_fn() -> int:
#    return 42
#
#
## This test fails prior to adding generic (ParamSpec and TypeVar) type annotations to the
## requires_modules decorator, as the revealed type is "Any".
#def test_mypy_type() -> None:
#    res = require_fn()
#    assert_type(res, int)
