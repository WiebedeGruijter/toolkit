from dataclasses import dataclass
from toolkit.defines.instrumentbase import InstrumentBase

@dataclass(frozen=True)
class ModelingSettings():
    """
    A data class to hold all settings for a given simulation run.
    """
    instrument: InstrumentBase
    exposure_time: float | None = None
    use_gpu: bool = False