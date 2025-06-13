import pytest
import numpy as np
from toolkit.defines.instrument import JWSTMiri, JWSTNirSpecIFU
from toolkit.defines.modelingsettings import ModelingSettings

def test_jwst_miri_pixel_layout():
    """
    Tests that the JWSTMiri instrument defines its FOV correctly.
    It should be treated as a single 1x1 pixel grid.
    """
    instrument = JWSTMiri()
    x_edges, y_edges = instrument.get_pixel_layout()
    
    # A 1x1 grid is defined by 2 edges on each axis
    assert x_edges.shape == (2,)
    assert y_edges.shape == (2,)
    
    np.testing.assert_allclose(x_edges, [-5.5, 5.5])
    np.testing.assert_allclose(y_edges, [-6.0, 6.0])

def test_jwst_nirspec_ifu_pixel_layout():
    """
    Tests that the JWSTNirSpecIFU instrument defines its 30x30 pixel
    grid correctly.
    """
    instrument = JWSTNirSpecIFU()
    x_edges, y_edges = instrument.get_pixel_layout()
    
    # A 30x30 grid is defined by 31 edges on each axis
    assert x_edges.shape == (31,)
    assert y_edges.shape == (31,)
    
    # Total FOV is 3.0 arcseconds, so edges run from -1.5 to 1.5
    np.testing.assert_allclose(x_edges, np.linspace(-1.5, 1.5, 31))
    np.testing.assert_allclose(y_edges, np.linspace(-1.5, 1.5, 31))

def test_modeling_settings_creation(miri_instrument):
    """
    Tests the basic creation of a ModelingSettings object.
    """
    settings = ModelingSettings(instrument=miri_instrument, exposure_time=100.0)
    assert settings.instrument == miri_instrument
    assert settings.exposure_time == 100.0