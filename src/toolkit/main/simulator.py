from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.main.simulatorbase import CubeSimulator
from toolkit.modeling.model_spectrum import apply_instrumental_effects
import xarray as xr

class InstrumentSimulator(CubeSimulator):
    """
    A simulator that models an instrument's response to a 3D source cube.

    This class can provide the full 3D "IFU-style" data cube, or it can
    provide the spatially-integrated "point-source-style" 1D spectrum.
    
    The noise is always calculated on a per-pixel basis for physical accuracy
    before any spatial integration is performed.
    """

    def get_flux_at_detector(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """
        Gets the clean (noise-free) flux as a 3D data cube (Wavelength, Y, X).
        
        This represents the ideal signal hitting each detector pixel.
        """
        return self._get_flux_on_detector_grid(modeling_settings)

    def get_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """
        Gets the observed flux, including instrumental noise, as a 3D data cube.
        
        This is the primary output, simulating a full IFU data cube.
        """
        # 1. Get the clean, noise-free data cube.
        clean_cube = self.get_flux_at_detector(modeling_settings)
        
        # 2. Apply instrumental effects to each pixel's spectrum. Note that this changes the shape of the array.
        noisy_cube_reordered = xr.apply_ufunc(
            apply_instrumental_effects,
            clean_cube,                      # First input arg (for flux_values)
            clean_cube.wavelength,           # Second input arg (for wavelength_values)
            input_core_dims=[['wavelength'], ['wavelength']], # Rules for each input
            output_core_dims=[['wavelength']],   # Rule for the output
            vectorize=True,
            keep_attrs=True, # It's good practice to keep attributes
            kwargs={'modeling_settings': modeling_settings} # Extra args for the function
        )

        # 3. Transpose the dimensions back to the original shape.
        return noisy_cube_reordered.transpose(*clean_cube.dims)
    
    def get_spatially_integrated_flux(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """
        Gets the clean (noise-free) flux, spatially integrated into a 1D spectrum.
        """
        clean_cube = self.get_flux_at_detector(modeling_settings)
        return clean_cube.sum(dim=['pix_x', 'pix_y'])

    def get_spatially_integrated_observed_spectrum(self, modeling_settings: ModelingSettings) -> xr.DataArray:
        """
        Gets the observed flux with noise, spatially integrated into a 1D spectrum.
        """
        noisy_cube = self.get_observed_spectrum(modeling_settings)
        return noisy_cube.sum(dim=['pix_x', 'pix_y'])