from enum import Enum


class MaterializeSourceMode(Enum):
    RECOMPUTE = 0
    USE_STORED = 1

    # Deprecated constants
    OFF = 0
    IF_PRESENT = 1
