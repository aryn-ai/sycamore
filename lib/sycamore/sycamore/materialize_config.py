from enum import Enum


class MaterializeSourceMode(Enum):
    """
    See DocSet.materialize for documentation on the semantics of these
    """

    RECOMPUTE = 0
    USE_STORED = 1

    # Deprecated constants
    OFF = 0
    IF_PRESENT = 1
