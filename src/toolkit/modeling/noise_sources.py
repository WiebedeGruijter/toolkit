from scipy.constants import c, h
import numpy as np
from toolkit.defines.modelingsettings import ModelingSettings

def poisson_noise(flux_at_detector, wavelength_nm, modeling_settings):
    """
    Applies Poisson shot noise to a flux signal.

    This function converts the physical flux (energy/s/area/wl) into the
    total number of photons collected by the instrument over the exposure time,
    applies Poisson noise to this count, and then converts the noisy count
    back into flux units.
    """
    telescope_area = np.pi*(modeling_settings.instrument.mirror_diameter/2)**2
    exposure_time = modeling_settings.exposure_time

    wavelength_m = wavelength_nm * 1e-9
    
    energy_per_photon = h * c / wavelength_m

    total_energy_collected = flux_at_detector * telescope_area * exposure_time

    n_photons = total_energy_collected / energy_per_photon
    noisy_n_photons = np.random.poisson(n_photons)

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