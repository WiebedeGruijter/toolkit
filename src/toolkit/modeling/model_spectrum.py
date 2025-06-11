from toolkit.defines.modelingsettings import ModelingSettings
from xarray import DataArray
from toolkit.modeling.noise_sources import poisson_noise, readout_noise, instrumental_broadening, linear_baseline_drift
import numpy as np

def convert_specific_intensity_to_flux(input_intensity: np.ndarray, modeling_settings: ModelingSettings):
    '''Convert specific intensity (J/s/m^2/nm/sr) to flux at the instrument (J/s/m^2/nm), taking into account distance to source.'''

    flux_at_detector = np.pi * (modeling_settings.source_radius/modeling_settings.distance_to_source)**2 * input_intensity
    return flux_at_detector

def calculate_1D_observed_spectrum(input_spectrum: DataArray, modeling_settings: ModelingSettings):
    """
    Applies simulated noise and instrumental effects to a 1D spectrum.
    """
    assert input_spectrum.ndim == 1, f"Input spectrum must be 1D, but got {input_spectrum.ndim} dimensions."

    input_intensity = input_spectrum.values # Numpy arrays are faster than DataArrays

    # Convert the specific intensity to flux at the detector
    flux_at_detector = convert_specific_intensity_to_flux(input_intensity=input_intensity, modeling_settings=modeling_settings)

    flux_at_detector = instrumental_broadening(flux_at_detector, modeling_settings)
    flux_at_detector = poisson_noise(flux_at_detector, input_spectrum.wavelength, modeling_settings)

    flux_at_detector += readout_noise(modeling_settings)
    flux_at_detector += linear_baseline_drift(modeling_settings)
    
    output_spectrum = input_spectrum.copy(data=flux_at_detector)
    return output_spectrum
