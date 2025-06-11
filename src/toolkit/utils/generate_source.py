import numpy as np
import xarray as xr
from astropy.constants import R_sun, R_earth
from scipy.constants import parsec, au
from toolkit.utils.blackbody import planck_law

def generate_data_cube(savepath: str, n_pix_xy=256, star_temp=6000, star_radius_rsun=1,
                       planet_temp=300, planet_radius_rearth=1,
                       distance_pc=10, sma_au=1):
    """
    Generate a 3D data cube with the specific intensity distribution (W m^-2 sr^-1 nm^-1)
    of a star and a planet as seen from Earth.

    Returns:
        xarray.DataArray: (wavelength [nm], y, x) intensity cube
    """

    # Radii & distances
    star_radius = star_radius_rsun * R_sun
    planet_radius = planet_radius_rearth * R_earth
    distance = distance_pc * parsec
    sma = sma_au * au

    # Blackbody spectra (in W·m⁻²·sr⁻¹·nm⁻¹)
    wavelength_nm, star_spectrum = planck_law(temperature_k=star_temp)
    _, planet_spectrum = planck_law(temperature_k=planet_temp)

    n_wave = len(wavelength_nm)

    # Angular resolution
    fov_rad = 3 * sma / distance
    pixel_scale_rad = fov_rad / n_pix_xy

    # Solid angles
    pixel_solid_angle = pixel_scale_rad**2
    star_solid_angle = np.pi * (star_radius.value / distance)**2
    planet_solid_angle = np.pi * (planet_radius.value / distance)**2

    # Filling factor = Ω_obj / Ω_pixel
    star_filling_factor = star_solid_angle / pixel_solid_angle
    planet_filling_factor = planet_solid_angle / pixel_solid_angle

    # Intensity per pixel
    I_pixel_star = star_spectrum * star_filling_factor  # W·m⁻²·sr⁻¹·nm⁻¹
    I_pixel_planet = planet_spectrum * planet_filling_factor

    # Allocate data cube
    data_cube = np.zeros((n_wave, n_pix_xy, n_pix_xy))

    # Star at center
    star_pos = [n_pix_xy // 2, n_pix_xy // 2]
    data_cube[:, star_pos[0], star_pos[1]] = I_pixel_star

    # Planet offset along x-axis
    planet_offset_pix = int((sma / distance) / pixel_scale_rad)
    planet_pos = [star_pos[0], star_pos[1] + planet_offset_pix]
    if 0 <= planet_pos[0] < n_pix_xy and 0 <= planet_pos[1] < n_pix_xy:
        data_cube[:, planet_pos[0], planet_pos[1]] = I_pixel_planet

    # Wrap as xarray
    data_xr = xr.DataArray(
        data_cube,
        dims=["wavelength", "y", "x"],
        coords={
            "wavelength": wavelength_nm,
            "x": np.arange(n_pix_xy),
            "y": np.arange(n_pix_xy)
        },
        name="specific_intensity",
        attrs={"units": "W m^-2 sr^-1 nm^-1", 'distance_pc': distance_pc}
    )
    data_xr.to_netcdf(savepath)
    return data_xr
