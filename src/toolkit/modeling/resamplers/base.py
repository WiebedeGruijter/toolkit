from abc import ABC, abstractmethod
import numpy as np
import xarray as xr

from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.utils.unit_conversions import ARCSEC_2_TO_STERADIAN

class ResamplerBase(ABC):
    """
    Abstract base class for resampling strategies. It handles common setup
    and defines the interface for all concrete resampler implementations.
    """
    def __init__(self, source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings):
        self.source_cube = source_cube
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.modeling_settings = modeling_settings
        self.instrument = modeling_settings.instrument

        # Pre-calculate common values
        y_coords, x_coords = self.source_cube.coords['y'].values, self.source_cube.coords['x'].values
        self.dx_source = abs(x_coords[1] - x_coords[0])
        
        source_pixel_solid_angle = (self.dx_source**2) * ARCSEC_2_TO_STERADIAN
        self.source_flux_map = self.source_cube * source_pixel_solid_angle

        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        self.x_source_flat = xx.ravel()
        self.y_source_flat = yy.ravel()

    @abstractmethod
    def resample(self) -> xr.DataArray:
        """
        The core resampling method that must be implemented by each backend.
        """
        pass

    def _create_final_cube(self, data: np.ndarray) -> xr.DataArray:
        """Helper to construct the final xarray.DataArray."""
        cube = xr.DataArray(
            data=data,
            dims=['wavelength', 'pix_y', 'pix_x'],
            coords={
                'wavelength': self.source_cube.coords['wavelength'],
                'pix_y': np.arange(len(self.y_edges) - 1),
                'pix_x': np.arange(len(self.x_edges) - 1)
            },
            attrs={"units": "W m^-2 nm^-1"}
        )
        return cube