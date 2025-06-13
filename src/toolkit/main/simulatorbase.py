from abc import ABC, abstractmethod
from toolkit.defines.modelingsettings import ModelingSettings
import xarray as xr
from toolkit.modeling.spatial_resampling import resample_source_to_instrument_grid
from toolkit.read_data.source import read_source_3D_cube

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

class CubeSimulator(SimulatorBase):
    """The common base class for all simulators that start with a 3D source cube."""
    def __init__(self, filepath: str):
        super().__init__()
        self.input_spectrum = read_source_3D_cube(filepath=filepath)

    def _get_flux_on_detector_grid(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """The common core: projects the source onto the instrument's detector grid."""
        instrument = modeling_settings.instrument
        x_edges, y_edges = instrument.get_pixel_layout()
        return resample_source_to_instrument_grid(
            source_cube=self.input_spectrum, x_edges=x_edges, y_edges=y_edges)