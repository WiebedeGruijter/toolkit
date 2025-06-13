from dataclasses import dataclass
from toolkit.defines.instrumentbase import InstrumentBase

@dataclass
class ModelingSettings():
    """
    A data class to hold all settings for a given simulation run.
    """
    instrument: InstrumentBase
    exposure_time: float | None = None