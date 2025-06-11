from dataclasses import dataclass
from toolkit.defines.instrumentbase import InstrumentBase

@dataclass
class ModelingSettings():
    
    instrument: InstrumentBase
    distance_to_source: float
    source_radius: float
    exposure_time: float | None = None