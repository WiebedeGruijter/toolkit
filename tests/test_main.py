import pytest
import xarray as xr
import numpy as np

def test_instrument_simulator_init(instrument_simulator):
    """
    Tests the initialization of the InstrumentSimulator.
    """
    assert hasattr(instrument_simulator, 'input_spectrum')
    assert isinstance(instrument_simulator.input_spectrum, xr.DataArray)

def test_get_clean_flux_at_detector(instrument_simulator, nirspec_settings):
    """
    Tests the calculation of the clean, noise-free flux cube.
    """
    clean_cube = instrument_simulator.get_clean_flux_at_detector(nirspec_settings)
    assert isinstance(clean_cube, xr.DataArray)
    assert clean_cube.dims == ('wavelength', 'pix_y', 'pix_x')
    assert clean_cube.shape == (50, 30, 30)

def test_get_observed_spectrum(instrument_simulator, nirspec_settings):
    """
    Tests the calculation of the noisy, observed flux cube.
    The output should be different from the clean input.
    """
    np.random.seed(42)
    clean_cube = instrument_simulator.get_clean_flux_at_detector(nirspec_settings)
    
    with pytest.warns(UserWarning): # Suppress warnings, can be removed once full instrument simulator is implemented
        noisy_cube = instrument_simulator.get_observed_spectrum(nirspec_settings)

    assert isinstance(noisy_cube, xr.DataArray)
    assert noisy_cube.shape == clean_cube.shape
    # Check that noise was actually added
    assert np.any(noisy_cube.values != clean_cube.values)

def test_get_spatially_integrated_flux(instrument_simulator, miri_settings):
    """
    Tests the spatial integration of the clean flux.
    """
    integrated_spectrum = instrument_simulator.get_spatially_integrated_flux(miri_settings)
    
    assert isinstance(integrated_spectrum, xr.DataArray)
    assert integrated_spectrum.dims == ('wavelength',)
    assert integrated_spectrum.shape == (50,)

    # Verify that it matches a manual summation
    clean_cube = instrument_simulator.get_clean_flux_at_detector(miri_settings)
    manual_sum = clean_cube.sum(dim=['pix_x', 'pix_y'])
    np.testing.assert_allclose(integrated_spectrum.values, manual_sum.values)

def test_get_spatially_integrated_observed_spectrum(instrument_simulator, nirspec_settings):
    """
    Tests the spatial integration of the noisy, observed flux.
    """
    with pytest.warns(UserWarning): # Suppress warnings
        np.random.seed(42)
        integrated_spectrum = instrument_simulator.get_spatially_integrated_observed_spectrum(nirspec_settings)
        
        # Verify that it matches a manual summation of the noisy cube
        np.random.seed(42) # The previous call to a random function advances the random seed generator, so we need to set it again
        noisy_cube = instrument_simulator.get_observed_spectrum(nirspec_settings)
        manual_sum = noisy_cube.sum(dim=['pix_x', 'pix_y'])
        
    assert isinstance(integrated_spectrum, xr.DataArray)
    assert integrated_spectrum.dims == ('wavelength',)
    np.testing.assert_allclose(integrated_spectrum.values, manual_sum.values)