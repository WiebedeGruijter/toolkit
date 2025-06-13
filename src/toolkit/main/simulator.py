from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.main.simulatorbase import CubeSimulator
from toolkit.modeling.model_spectrum import apply_instrumental_effects
import xarray as xr

class PointSourceSimulator(CubeSimulator):
    """A simulator that integrates all flux from a 3D cube into a single 1D spectrum."""

    def get_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        flux_per_pixel_cube = self._get_flux_on_detector_grid(modeling_settings)
        total_flux_1d = flux_per_pixel_cube.sum(dim=['pix_x', 'pix_y'])
        return total_flux_1d

    def get_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        # 1. Get the clean, noise-free 1D DataArray.
        clean_spectrum = self.get_flux_at_detector(modeling_settings)
        
        # 2. Extract numpy arrays to pass to the core function.
        flux_vals = clean_spectrum.values
        wave_vals = clean_spectrum.wavelength.values
        
        # 3. Apply instrumental effects.
        noisy_flux_vals = apply_instrumental_effects(flux_vals, wave_vals, modeling_settings)
        
        # 4. Wrap the result back into a DataArray.
        return clean_spectrum.copy(data=noisy_flux_vals)

class IFUSimulator(CubeSimulator):
    """A simulator for an IFU, which preserves spatial information."""

    def get_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        return self._get_flux_on_detector_grid(modeling_settings)

    def get_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        # 1. Get the clean, noise-free data cube.
        clean_cube = self.get_flux_at_detector(modeling_settings)
        
        # 2. Apply instrumental effects using apply_ufunc.
        return xr.apply_ufunc(
            apply_instrumental_effects,
            clean_cube,                     # First input arg (for flux_values)
            clean_cube.wavelength,          # Second input arg (for wavelength_values)
            input_core_dims=[['wavelength'], ['wavelength']], # Rules for each input
            output_core_dims=[['wavelength']],                 # Rule for the output
            vectorize=True,
            kwargs={'modeling_settings': modeling_settings} # Extra args for the function
        )