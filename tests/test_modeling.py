import pytest
import numpy as np
from toolkit.modeling.noise_sources import (
    poisson_noise, readout_noise, calculate_psf, linear_baseline_drift)
from toolkit.modeling.model_spectrum import apply_instrumental_effects
from toolkit.utils.unit_conversions import ARCSEC_2_TO_STERADIAN

def test_resample_to_miri(instrument_simulator, miri_settings):
    """
    Tests resampling a source cube onto the MIRI 1x1 grid.
    """
    resampled_cube = instrument_simulator._get_clean_flux_on_detector_grid(miri_settings)
    assert resampled_cube.dims == ('wavelength', 'pix_y', 'pix_x')
    assert resampled_cube.shape[1:] == (1, 1) # MIRI is 1x1

def test_resample_to_nirspec(instrument_simulator, nirspec_settings):
    """
    Tests resampling a source cube onto the NIRSpec 30x30 grid.
    """
    resampled_cube = instrument_simulator._get_clean_flux_on_detector_grid(nirspec_settings)
    assert resampled_cube.dims == ('wavelength', 'pix_y', 'pix_x')
    assert resampled_cube.shape[1:] == (30, 30) # NIRSpec is 30x30

def test_flux_conservation(instrument_simulator, miri_settings):
    """
    Tests that the total flux is conserved during resampling.
    """
    source_cube = instrument_simulator.input_spectrum
    
    # Calculate total input flux (accounting for source pixel area)
    dx = abs(source_cube.x[1] - source_cube.x[0]).item()
    dy = abs(source_cube.y[1] - source_cube.y[0]).item()
    source_pixel_area = dx * dy * ARCSEC_2_TO_STERADIAN
    total_input_flux = source_cube.sum(dim=['x', 'y']) * source_pixel_area

    # Resample to instrument grid (MIRI captures the whole FOV)
    resampled_cube = instrument_simulator._get_clean_flux_on_detector_grid(miri_settings)
    
    # The resampled cube values are already total flux per pixel, so we just sum
    total_output_flux = resampled_cube.sum(dim=['pix_x', 'pix_y'])
    
    # The values should be very close (allowing for float precision)
    np.testing.assert_allclose(total_input_flux.values, total_output_flux.values, rtol=1e-6)

def test_poisson_noise(miri_settings):
    """
    Tests the poisson_noise function.
    """
    np.random.seed(42) # for reproducibility
    
    # Test case 1: Zero flux in, zero flux out
    flux_in_zero = np.zeros(100)
    wl_in = np.linspace(400, 500, 100)
    flux_out_zero = poisson_noise(flux_in_zero, wl_in, miri_settings)
    assert np.all(flux_out_zero == 0)

    # Test case 2: Constant high flux
    # We expect the noisy flux to be different from the input
    flux_in_const = np.full(100, 1e-15)
    noisy_flux = poisson_noise(flux_in_const, wl_in, miri_settings)
    assert np.any(noisy_flux != flux_in_const)
    
    # The mean should be close to the original signal
    np.testing.assert_allclose(np.mean(noisy_flux), np.mean(flux_in_const), rtol=0.1)

def test_placeholder_noise_functions(miri_settings):
    """
    Tests that the unimplemented noise functions return 0 and issue a warning.
    """
    with pytest.warns(UserWarning, match='Readout noise not implemented'):
        assert readout_noise(miri_settings) == 0
        
    with pytest.warns(UserWarning, match='Instrumental broadening not implemented'):
        flux = np.ones(10)
        assert np.all(calculate_psf(flux, miri_settings) == flux)
        
    with pytest.warns(UserWarning, match='Baseline drift not implemented'):
        assert linear_baseline_drift(miri_settings) == 0

def test_apply_instrumental_effects(miri_settings):
    """
    Tests the top-level function for applying all instrumental effects.
    """
    np.random.seed(42)
    flux_in = np.ones(100) * 1e-15
    wl_in = np.linspace(400, 500, 100)

    with pytest.warns(UserWarning): # Suppress warnings from placeholder functions
        noisy_flux = apply_instrumental_effects(flux_in, wl_in, miri_settings)

    assert noisy_flux.shape == flux_in.shape
    assert np.any(noisy_flux != flux_in)