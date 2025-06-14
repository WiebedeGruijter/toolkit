import pytest
import xarray as xr
from pathlib import Path
from toolkit.read_data.source import read_source_3D_cube, read_point_source

def test_read_source_3d_cube(datacube_filepath: str):
    """
    Tests reading a 3D source cube from a NetCDF file.
    """
    data_array = read_source_3D_cube(datacube_filepath)
    
    assert isinstance(data_array, xr.DataArray)
    assert data_array.dims == ('wavelength', 'y', 'x')
    assert 'wavelength' in data_array.coords
    assert 'x' in data_array.coords
    assert 'y' in data_array.coords
    assert data_array.name == 'specific_intensity'

def test_read_point_source(point_source_filepaths: dict[str, Path]):
    """
    Tests reading a point source from a 2-column text file.
    """
    data_array = read_point_source(point_source_filepaths['T5000'])
    
    assert isinstance(data_array, xr.DataArray)
    assert data_array.dims == ('wavelength',)
    assert data_array.name == 'intensity'
    assert data_array.attrs['units'] == 'J/s/m^2/nm/sr'
    assert data_array.ndim == 1

def test_read_point_source_value_error(tmp_path: Path):
    """
    Tests that read_point_source raises a ValueError for a file
    with the wrong number of columns.
    """
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bad_file.txt"
    p.write_text("1 2 3\n4 5 6") # Write a 3-column file

    with pytest.raises(ValueError, match="File must have two columns"):
        read_point_source(p)