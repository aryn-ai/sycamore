from sycamore.utils.import_utils import requires_modules
from sycamore.utils.time_trace import LogTime
import fasteners
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import DeformableDetrForObjectDetection

_DETR_LOCK_FILE = f"{Path.home()}/.cache/Aryn-Detr.lock"


@requires_modules("transformers", "local_inference")
def load_deformable_detr(model_name_or_path) -> "DeformableDetrForObjectDetection":
    """Load deformable detr without getting concurrency issues in
    jit-ing the deformable attention kernel.

    Refactored out of:
    https://github.com/aryn-ai/sycamore/blob/7e6b62639ce9b8f63d56cb35a32837d1c97e711e/lib/sycamore/sycamore/transforms/detr_partitioner.py#L686
    """
    from sycamore.utils.pytorch_dir import get_pytorch_build_directory

    with fasteners.InterProcessLock(_DETR_LOCK_FILE):
        lockfile = Path(get_pytorch_build_directory("MultiScaleDeformableAttention", False)) / "lock"
        lockfile.unlink(missing_ok=True)

        from transformers import DeformableDetrForObjectDetection

        LogTime("loading_model", point=True)
        with LogTime("loading_model", log_start=True):
            model = DeformableDetrForObjectDetection.from_pretrained(model_name_or_path)
    return model
