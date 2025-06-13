from scipy.constants import c, h
import numpy as np
from toolkit.defines.modelingsettings import ModelingSettings

def poisson_noise(flux_at_detector, wavelength_nm, modeling_settings):
    """
    Applies Poisson shot noise to a flux signal.

    This function converts the physical flux (energy/s/area/wl) into the
    total number of photons collected. For a large number of photons, it uses
    a computationally stable Normal distribution to approximate the Poisson
    distribution. For a small number, it uses the exact Poisson distribution.
    """
    telescope_area = np.pi * (modeling_settings.instrument.mirror_diameter / 2)**2
    exposure_time = modeling_settings.exposure_time

    wavelength_m = wavelength_nm * 1e-9
    
    energy_per_photon = h * c / wavelength_m

    total_energy_collected = flux_at_detector * telescope_area * exposure_time

    n_photons = total_energy_collected / energy_per_photon
    
    # Handle large photon counts with a Gaussian approximation (np.random.poisson caanot handle large numbers)
    noisy_n_photons = np.zeros_like(n_photons, dtype=float)

    # Set a threshold for when to switch from Poisson to Gaussian statistics.
    # A value of 1e6 is very safe; the approximation is excellent here.
    GAUSSIAN_APPROXIMATION_THRESHOLD = 1e6

    # Create masks for the two regimes
    large_n_mask = n_photons > GAUSSIAN_APPROXIMATION_THRESHOLD
    small_n_mask = ~large_n_mask

    # 1. For large photon counts, use the Normal approximation.
    #    Mean = n_photons, Standard Deviation = sqrt(n_photons)
    if np.any(large_n_mask):
        mean = n_photons[large_n_mask]
        std_dev = np.sqrt(mean)
        noisy_n_photons[large_n_mask] = np.random.normal(loc=mean, scale=std_dev)

    # 2. For small photon counts, use the exact Poisson distribution.
    if np.any(small_n_mask):
        # np.random.poisson can't handle an input array with zeros if other values are large
        # so we apply it only to the relevant subset.
        photons_for_poisson = n_photons[small_n_mask]
        noisy_n_photons[small_n_mask] = np.random.poisson(photons_for_poisson)

    # The Normal distribution can produce negative numbers, which is unphysical.
    # Set any negative photon counts to zero.
    noisy_n_photons[noisy_n_photons < 0] = 0.0

    noisy_total_energy = noisy_n_photons * energy_per_photon
    noisy_flux = noisy_total_energy / (telescope_area * exposure_time)
    
    return noisy_flux

def readout_noise( modeling_settings: ModelingSettings):
    import warnings
    warnings.warn('Readout noise not implemented yet, defaulting to 0')
    return 0

def instrumental_broadening(flux: np.ndarray, modeling_settings: ModelingSettings):
    import warnings
    warnings.warn('Instrumental broadening not implemented yet, defaulting to no broadening')
    return flux

def linear_baseline_drift(modeling_settings: ModelingSettings):
    import warnings
    warnings.warn('Baseline drift not implemented yet, defaulting to 0')
    return 0