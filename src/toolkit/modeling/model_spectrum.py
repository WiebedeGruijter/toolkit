from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.modeling.noise_sources import (
    poisson_noise, readout_noise, calculate_psf, linear_baseline_drift)
import numpy as np

def apply_instrumental_effects(flux_values: np.ndarray, wavelength_values: np.ndarray, modeling_settings: ModelingSettings) -> np.ndarray:
    """
    Applies simulated noise and instrumental effects to a 1D numpy array of flux values.
    This is a "core" function designed to be compatible with xr.apply_ufunc.

    Args:
        flux_values (np.ndarray): A 1D array of the source flux at the detector.
        wavelength_values (np.ndarray): A 1D array of corresponding wavelengths.
        modeling_settings (ModelingSettings): The settings for the simulation.

    Returns:
        np.ndarray: The final observed flux values with noise and other effects.
    """
    assert flux_values.ndim == 1, f"Input flux array must be 1D, but got {flux_values.ndim}."
    
    noisy_flux = flux_values.copy()

    # Apply effects in sequence
    noisy_flux = calculate_psf(noisy_flux, modeling_settings)
    noisy_flux = poisson_noise(noisy_flux, wavelength_values, modeling_settings)
    noisy_flux += readout_noise(modeling_settings)
    noisy_flux += linear_baseline_drift(modeling_settings)
    
    return noisy_flux