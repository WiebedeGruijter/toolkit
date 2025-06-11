import numpy as np
import xarray as xr

def read_point_source(filepath):
    '''
    Initializes wavelength and intensity from a 2-column txt file.
    Column 1: wavelength (nm)
    Column 2: intensity (J/s/m²/nm/sr)
    '''
    raw_data = np.loadtxt(filepath, comments='#')
    if raw_data.shape[1] != 2:
        raise ValueError("File must have two columns: wavelength, intensity")

    wavelength = raw_data[:, 0]
    intensity = raw_data[:, 1]

    data_array = xr.DataArray(
        data=intensity,
        coords={'wavelength': wavelength},
        dims=['wavelength'],
        name='intensity',
        attrs={'units': 'J/s/m^2/nm/sr'}
    )
    data_array.wavelength.attrs['units'] = 'nm'
    return data_array
        
def read_source_3D_cube(filepath):
    '''
    Reads in a source from a netcdf file and creates a 3D datacube.
    '''
    dataset = xr.open_dataset(filepath)
    data_array = dataset['specific_intensity']
    return data_array
