# src\toolkit\main\simulator.py

from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.main.simulatorbase import SimulatorBase
from toolkit.modeling.resampling import resample_source_to_instrument_grid
from toolkit.modeling.model_spectrum import apply_instrumental_effects
from toolkit.read_data.source import read_source_3D_cube
import xarray as xr

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

class PointSourceSimulator(CubeSimulator):
    """A simulator that integrates all flux from a 3D cube into a single 1D spectrum."""

    def get_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """Returns the clean, noise-free 1D spectrum."""
        flux_per_pixel_cube = self._get_flux_on_detector_grid(modeling_settings)
        # Integrate over all detector pixels to get the total flux.
        total_flux_1d = flux_per_pixel_cube.sum(dim=['pix_x', 'pix_y'])
        return total_flux_1d

    def get_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """Returns the final spectrum with noise."""
        # 1. Get the clean, noise-free spectrum.
        clean_spectrum = self.get_flux_at_detector(modeling_settings)
        # 2. Apply instrumental effects to it.
        return apply_instrumental_effects(clean_spectrum, modeling_settings)

class IFUSimulator(CubeSimulator):
    """A simulator for an IFU, which preserves spatial information."""

    def get_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """Returns the clean, noise-free 3D data cube."""
        return self._get_flux_on_detector_grid(modeling_settings)

    def get_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """Returns the final 3D cube with noise applied to each pixel."""
        # 1. Get the clean, noise-free data cube.
        clean_cube = self.get_flux_at_detector(modeling_settings)
        # 2. Apply instrumental effects to each pixel's spectrum individually.
        return xr.apply_ufunc(
            apply_instrumental_effects, clean_cube,
            input_core_dims=[['wavelength']], output_core_dims=[['wavelength']],
            exclude_dims=set(('wavelength',)), vectorize=True,
            kwargs={'modeling_settings': modeling_settings})