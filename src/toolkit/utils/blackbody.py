import numpy as np
from scipy.constants import h, c, k
from pathlib import Path

def planck_law(savepath:Path | None=None, wl_min_nm=380, wl_max_nm=640, nbins=500, temperature_k=5000, savefile=False):
    """
    Calculates the spectral radiance of a blackbody using Planck's Law.

    Args:
        wl_min_nm (float, optional): The minimum wavelength in nanometers. 
        wl_max_nm (float, optional): The maximum wavelength in nanometers. 
        nbins (int, optional): The number of wavelength bins to generate. 
        temperature_k (float, optional): The temperature of the blackbody in Kelvin. 

    Returns:
        np.ndarray: An array of spectral radiance values in units of 
                    J/s/m^2/nm/sr (Watts per square meter per nanometer 
                    per steradian).
    """
    # Convert wavelengths from nanometers to meters for the calculation
    wavelengths_nm = np.linspace(wl_min_nm, wl_max_nm, nbins)
    wavelengths_m = wavelengths_nm * 1e-9

    # Planck's Law formula
    # The exponential term can be very large, so we handle potential overflows.
    # We calculate the spectral radiance per unit wavelength in meters first.
    exponent = (h * c) / (wavelengths_m * k * temperature_k)
    
    # Avoid overflow for very small wavelengths or low temperatures
    # by setting a maximum sensible value for the exponent.
    # If exp(exponent) would be huge, the denominator is essentially exp(exponent).
    # If exp(exponent) is close to 1, the (exp(x) - 1) is important.
    # A value of ~709 is where np.exp overflows to infinity for float64.
    radiance_per_meter = np.zeros_like(wavelengths_m)
    mask = exponent < 709 # Only calculate for values that won't overflow
    
    numerator = 2.0 * h * c**2
    denominator = (wavelengths_m[mask]**5) * (np.exp(exponent[mask]) - 1.0)
    
    radiance_per_meter[mask] = numerator / denominator

    # Convert units from J/s/m^2/m/sr to J/s/m^2/nm/sr by dividing by 1e9
    radiance_per_nm = radiance_per_meter / 1e9

    if savefile:
        data = np.column_stack((wavelengths_nm, radiance_per_nm))
        header_text = 'wavelength[nm]    specific_intensity[J/s/m^2/nm/sr]'
        np.savetxt(savepath / f'blackbody_T{temperature_k}.txt', data, delimiter='\t', fmt='%.2f', header=header_text)

    return wavelengths_nm, radiance_per_nm
