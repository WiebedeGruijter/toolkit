from toolkit.defines.modelingsettings import ModelingSettings
from xarray import DataArray
from toolkit.modeling.noise_sources import (
    poisson_noise,
    readout_noise,
    instrumental_broadening,
    linear_baseline_drift
)
import numpy as np

def apply_instrumental_effects(flux_spectrum: DataArray, modeling_settings: ModelingSettings) -> DataArray:
    """
    Applies simulated noise and instrumental effects to a 1D flux spectrum.
    This function can be applied to each pixel's spectrum in an IFU cube.

    Args:
        flux_spectrum (DataArray): A 1D DataArray of the source flux at the detector.
        modeling_settings (ModelingSettings): The settings for the simulation.

    Returns:
        DataArray: The final observed spectrum with noise and other effects.
    """
    assert flux_spectrum.ndim == 1, f"Input flux spectrum must be 1D, but got {flux_spectrum.ndim} dimensions."

    flux_values = flux_spectrum.values
    wavelength_values = flux_spectrum.coords['wavelength'].values

    # Apply effects in sequence
    flux_values = instrumental_broadening(flux_values, modeling_settings)
    flux_values = poisson_noise(flux_values, wavelength_values, modeling_settings)
    flux_values += readout_noise(modeling_settings)
    flux_values += linear_baseline_drift(modeling_settings)
    
    # Return a new DataArray with the noisy data
    output_spectrum = flux_spectrum.copy(data=flux_values)
    return output_spectrum