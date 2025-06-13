# src\toolkit\main\simulatorbase.py

from abc import ABC, abstractmethod
from toolkit.defines.modelingsettings import ModelingSettings
import xarray as xr

class SimulatorBase(ABC):
    """An abstract base class for all instrument simulators."""

    def __init__(self):
        pass

    @abstractmethod
    def get_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """
        Calculates the ideal, noise-free signal at the detector.
        This should return the "clean" data before noise is added.
        """
        pass

    @abstractmethod
    def get_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """
        Calculates the final "observed" signal, including instrumental effects.
        """
        pass