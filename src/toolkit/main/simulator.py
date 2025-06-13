from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.main.simulatorbase import CubeSimulator
from toolkit.modeling.model_spectrum import apply_instrumental_effects
import xarray as xr

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