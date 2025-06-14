import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.convolution import AiryDisk2DKernel, convolve
from multiprocessing import Pool
from toolkit.utils.unit_conversions import ARCSEC_2_TO_STERADIAN
from toolkit.defines.modelingsettings import ModelingSettings

def _apply_psf(image_slice: np.ndarray, wavelength_m: float, mirror_diameter: float, pixel_scale_arcsec: float) -> np.ndarray:
    """
    Applies a wavelength-dependent Airy disk PSF to a 2D image slice.

    This helper function calculates the size of the PSF for the given
    parameters and convolves it with the input image.

    Args:
        image_slice (np.ndarray): The 2D numpy array representing the image.
        wavelength_m (float): The wavelength in meters.
        mirror_diameter (float): The telescope mirror diameter in meters.
        pixel_scale_arcsec (float): The pixel scale of the input image in arcsec/pixel.

    Returns:
        np.ndarray: The image slice convolved with the appropriate PSF.
    """
    # The angular radius of the first null of the Airy disk is theta = 1.22 * lambda / D.
    first_null_rad = 1.22 * wavelength_m / mirror_diameter
    first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
    first_null_pix = first_null_arcsec / pixel_scale_arcsec

    # Only perform convolution if the PSF is larger than a small fraction of a pixel.
    if first_null_pix > 0.1:
        # Create an AiryDisk2DKernel from astropy.
        psf_kernel = AiryDisk2DKernel(first_null_pix)

        # Convolve using astropy's convolution function.
        convolved_slice = convolve(
            array=image_slice,
            kernel=psf_kernel,
            boundary='extend',
            normalize_kernel=True
        )
        return convolved_slice
    else:
        # If the PSF is very small, return the original image to save computation.
        return image_slice


def resample_source_to_instrument_grid(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings) -> xr.DataArray:
    """
    Resamples a source cube onto an instrument grid, applying a PSF convolution
    at each wavelength before binning. The PSF application is parallelized across
    all available CPU cores using Python's multiprocessing module.
    """

    # 1. Get coordinate axes from the source cube.
    y_coords = source_cube.coords['y'].values
    x_coords = source_cube.coords['x'].values

    # 2. Create 2D coordinate grids for flattening later.
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    x_source_flat = xx.ravel()
    y_source_flat = yy.ravel()

    # 3. Convert source intensity to flux per source pixel.
    dx_source = abs(x_coords[1] - x_coords[0])
    dy_source = abs(y_coords[1] - y_coords[0])
    source_pixel_solid_angle = (dx_source * dy_source) * ARCSEC_2_TO_STERADIAN
    source_flux_map = source_cube * source_pixel_solid_angle

    # 4. Prepare arguments for the parallel execution of _apply_psf.
    mirror_diameter = modeling_settings.instrument.mirror_diameter
    args_for_starmap = [
        (
            wl_slice.values,
            wl_slice.wavelength.item() * 1e-9,
            mirror_diameter,
            dx_source
        ) for wl_slice in source_flux_map
    ]

    # 5. Apply the PSF to all slices in parallel using a multiprocessing Pool.
    print("Applying PSF convolution to all wavelength slices in parallel...")
    with Pool() as pool:
        convolved_slices = pool.starmap(_apply_psf, args_for_starmap)
    print("...PSF convolution complete.")


    # 6. Bin the convolved slices onto the detector grid. This part is fast.
    binned_flux_list = []
    for convolved_slice in convolved_slices:
        statistic, _, _, _ = binned_statistic_2d(
            x=x_source_flat,
            y=y_source_flat,
            values=convolved_slice.ravel(),
            statistic='sum',
            bins=[x_edges, y_edges]
        )
        binned_flux_list.append(statistic.T)

    # 7. Create the final DataArray for the instrument cube.
    flux_cube = xr.DataArray(
        data=np.array(binned_flux_list),
        dims=['wavelength', 'pix_y', 'pix_x'],
        coords={
            'wavelength': source_cube.coords['wavelength'],
            'pix_y': np.arange(len(y_edges) - 1),
            'pix_x': np.arange(len(x_edges) - 1)
        },
        attrs={"units": "W m^-2 nm^-1"}
    )
    return flux_cube