from abc import ABC, abstractmethod
from toolkit.defines.modelingsettings import ModelingSettings
import xarray as xr
from toolkit.modeling.spatial_resampling import resample_source_to_instrument_grid
from toolkit.read_data.source import read_source_3D_cube
from functools import lru_cache

class SimulatorBase(ABC):
    """An abstract base class for all instrument simulators."""

    def __init__(self):
        pass

    @abstractmethod
    def get_clean_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
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

class CubeSimulator(SimulatorBase):
    """The common base class for all simulators that start with a 3D source cube."""
    def __init__(self, filepath: str):
        super().__init__()
        self.input_spectrum = read_source_3D_cube(filepath=filepath)

    @lru_cache(maxsize=None)
    def _get_clean_flux_on_detector_grid(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """The common core: projects the source onto the instrument's detector grid, without any noise."""
        x_edges, y_edges = modeling_settings.instrument.get_pixel_layout()
        
        # This call is now cleaner and points to the refactored function
        return resample_source_to_instrument_grid(
            source_cube=self.input_spectrum, 
            x_edges=x_edges, 
            y_edges=y_edges, 
            modeling_settings=modeling_settings
        )