import numpy as np
import xarray as xr
from pathlib import Path
from toolkit.utils.blackbody import planck_law
from toolkit.utils.generate_source import generate_data_cube

def test_planck_law_output():
    """
    Tests the output of the planck_law function.
    """
    wl, radiance = planck_law(temperature_k=5000)
    assert wl.shape == (500,)
    assert radiance.shape == (500,)
    
    # A hotter blackbody should be brighter at all wavelengths
    _, radiance_hotter = planck_law(temperature_k=6000)
    assert np.all(radiance_hotter > radiance)

def test_planck_law_savefile(tmp_path: Path):
    """
    Tests the file saving functionality of the planck_law function.
    """
    save_dir = tmp_path / "test_output"
    save_dir.mkdir()
    
    planck_law(savepath=save_dir, temperature_k=3000, savefile=True)
    
    expected_file = save_dir / 'blackbody_T3000.txt'
    assert expected_file.is_file()
    
    # Check content
    data = np.loadtxt(expected_file, comments='#')
    assert data.shape[1] == 2

def test_generate_data_cube():
    """
    Tests the generation of a 3D source data cube.
    """
    data_cube = generate_data_cube(n_pix_xy=64, star_temp=5000, planet_temp=500)
    
    assert isinstance(data_cube, xr.DataArray)
    assert data_cube.dims == ('wavelength', 'y', 'x')
    assert data_cube.shape == (500, 64, 64)
    assert 'pixel_scale_arcsec' in data_cube.attrs

    # Check that the star is placed at the center pixel
    center_idx = 64 // 2 
    # Note: linspace with an even number of points doesn't have a true zero,
    # so we check the two center pixels.
    star_flux = data_cube[:, center_idx-1:center_idx+1, center_idx-1:center_idx+1].sum().item()
    assert star_flux > 0