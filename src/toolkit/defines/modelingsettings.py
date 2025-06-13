from dataclasses import dataclass
from toolkit.defines.instrumentbase import InstrumentBase

@dataclass
class ModelingSettings():
    
    instrument: InstrumentBase
    source_radius: float
    exposure_time: float | None = None