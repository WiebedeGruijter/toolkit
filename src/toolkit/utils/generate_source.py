import numpy as np
import xarray as xr
from astropy.constants import R_sun, R_earth
from scipy.constants import parsec, au
from toolkit.utils.blackbody import planck_law

# Conversion from radians to arcseconds
RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0

def generate_data_cube(savepath: str | None=None, n_pix_xy=256, star_temp=6000, star_radius_rsun=1,
                     planet_temp=300, planet_radius_rearth=1,
                     distance_pc=10, sma_au=1):
    """
    Generate a 3D data cube with specific intensity, using proper angular coordinates.
    """
    # --- Radii & distances ---
    star_radius = star_radius_rsun * R_sun.value
    planet_radius = planet_radius_rearth * R_earth.value
    distance = distance_pc * parsec
    sma = sma_au * au

    # --- Blackbody spectra ---
    wavelength_nm, star_spectrum = planck_law(temperature_k=star_temp)
    _, planet_spectrum = planck_law(temperature_k=planet_temp)
    n_wave = len(wavelength_nm)

    # --- Angular Grid Calculation ---
    fov_rad = 3 * sma / distance
    pixel_scale_rad = fov_rad / n_pix_xy
    pixel_scale_arcsec = pixel_scale_rad * RAD_TO_ARCSEC
    
    half_fov_arcsec = (fov_rad * RAD_TO_ARCSEC) / 2
    x_coords_arcsec = np.linspace(-half_fov_arcsec, half_fov_arcsec, n_pix_xy)
    y_coords_arcsec = np.linspace(-half_fov_arcsec, half_fov_arcsec, n_pix_xy)

    # --- Intensity Calculation ---
    pixel_solid_angle_sr = pixel_scale_rad**2
    star_solid_angle_sr = np.pi * (star_radius / distance)**2
    planet_solid_angle_sr = np.pi * (planet_radius / distance)**2

    star_filling_factor = star_solid_angle_sr / pixel_solid_angle_sr
    planet_filling_factor = planet_solid_angle_sr / pixel_solid_angle_sr

    I_pixel_star = star_spectrum * star_filling_factor
    I_pixel_planet = planet_spectrum * planet_filling_factor

    # --- Place objects in the data cube ---
    data_cube = np.zeros((n_wave, n_pix_xy, n_pix_xy))
    star_pos_idx = (np.abs(y_coords_arcsec).argmin(), np.abs(x_coords_arcsec).argmin())
    data_cube[:, star_pos_idx[0], star_pos_idx[1]] = I_pixel_star
    planet_angular_sep_arcsec = (sma / distance) * RAD_TO_ARCSEC
    planet_pos_idx = (np.abs(y_coords_arcsec).argmin(), np.abs(x_coords_arcsec - planet_angular_sep_arcsec).argmin())
    data_cube[:, planet_pos_idx[0], planet_pos_idx[1]] = I_pixel_planet

    # --- Create xarray.DataArray with correct coordinates ---
    data_xr = xr.DataArray(
        data_cube,
        dims=["wavelength", "y", "x"],
        coords={"wavelength": wavelength_nm, "y": y_coords_arcsec, "x": x_coords_arcsec},
        name="specific_intensity",
        attrs={"units": "W m^-2 sr^-1 nm^-1", "pixel_scale_arcsec": pixel_scale_arcsec}
    )
    data_xr.x.attrs['units'] = 'arcsec'
    data_xr.y.attrs['units'] = 'arcsec'
    data_xr.wavelength.attrs['units'] = 'nm'

    if savepath is not None:
        print(f'File saved as {savepath}')

        # Define the compression settings
        encoding_settings = {
            data_xr.name: {  # Use data_xr.name to be robust
                'zlib': True,
                'complevel': 5
            }
        }
        
        # Save the file with the specified encoding
        data_xr.to_netcdf(savepath, encoding=encoding_settings)
    return data_xr