import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.convolution import AiryDisk2DKernel, convolve
from toolkit.utils.unit_conversions import ARCSEC_2_TO_STERADIAN
from toolkit.defines.modelingsettings import ModelingSettings

# Retain the original, unchanged PSF helper function
def _apply_psf(image_slice: np.ndarray, wavelength_m: float, mirror_diameter: float, pixel_scale_arcsec: float) -> np.ndarray:
    """Applies a wavelength-dependent Airy disk PSF to a 2D image slice."""
    first_null_rad = 1.22 * wavelength_m / mirror_diameter
    first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
    first_null_pix = first_null_arcsec / pixel_scale_arcsec

    if first_null_pix > 0.1:
        psf_kernel = AiryDisk2DKernel(first_null_pix)
        convolved_slice = convolve(
            array=image_slice,
            kernel=psf_kernel,
            boundary='extend',
            normalize_kernel=True
        )
        return convolved_slice
    else:
        return image_slice

def _prepare_source_coordinates(source_cube: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates flattened 2D coordinate grids from the source cube's x and y axes.
    """
    y_coords = source_cube.coords['y'].values
    x_coords = source_cube.coords['x'].values
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    return xx.ravel(), yy.ravel()


def _psf_convolve_and_rebin_slice(
    image_slice: np.ndarray,
    wavelength_nm: float,
    x_source_flat: np.ndarray,
    y_source_flat: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mirror_diameter: float,
    pixel_scale_arcsec: float
) -> np.ndarray:
    """
    Core operation for apply_ufunc: Convolves a single 2D slice with a PSF
    and rebins it to the instrument grid.
    """
    # 1. Apply the PSF convolution for the given wavelength
    convolved_slice = _apply_psf(
        image_slice=image_slice,
        wavelength_m=wavelength_nm * 1e-9,  # Convert nm to m
        mirror_diameter=mirror_diameter,
        pixel_scale_arcsec=pixel_scale_arcsec
    )

    # 2. Bin the convolved slice onto the detector grid
    statistic, _, _, _ = binned_statistic_2d(
        x=x_source_flat,
        y=y_source_flat,
        values=convolved_slice.ravel(),
        statistic='sum',
        bins=[x_edges, y_edges]
    )
    return statistic.T

def resample_source_to_instrument_grid(
    source_cube: xr.DataArray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    modeling_settings: ModelingSettings
) -> xr.DataArray:
    """
    Resamples a source cube onto an instrument grid using a vectorized approach.

    This function applies a wavelength-dependent PSF convolution and then
    bins the data onto the new grid. The operation is vectorized over the
    'wavelength' dimension using xarray.apply_ufunc for high performance.
    """
    print("Applying vectorized PSF convolution and rebinning...")
    
    # 1. Prepare data for resampling
    dx_source = abs(source_cube.coords['x'].values[1] - source_cube.coords['x'].values[0])
    source_pixel_solid_angle = (dx_source**2) * ARCSEC_2_TO_STERADIAN
    source_flux_map = source_cube * source_pixel_solid_angle
    x_source_flat, y_source_flat = _prepare_source_coordinates(source_cube)

    # 2. Apply the core function vectorially across all wavelength slices
    flux_cube = xr.apply_ufunc(
        _psf_convolve_and_rebin_slice,  # The clean, module-level helper function
        source_flux_map,
        source_flux_map.wavelength,
        input_core_dims=[["y", "x"], []],
        output_core_dims=[["pix_y", "pix_x"]],
        exclude_dims=set(("y", "x")),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[source_flux_map.dtype],
        kwargs={
            "x_source_flat": x_source_flat,
            "y_source_flat": y_source_flat,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "mirror_diameter": modeling_settings.instrument.mirror_diameter,
            "pixel_scale_arcsec": dx_source,
        }
    )

    print("...Vectorized processing complete.")

    # 3. Assign final coordinates and attributes to the output DataArray
    final_cube = flux_cube.assign_coords(
        pix_y=np.arange(len(y_edges) - 1),
        pix_x=np.arange(len(x_edges) - 1)
    )
    final_cube.attrs["units"] = "W m^-2 nm^-1"
    
    return final_cube