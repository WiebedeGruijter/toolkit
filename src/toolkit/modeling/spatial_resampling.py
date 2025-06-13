import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
from toolkit.utils.unit_conversions import ARCSEC_2_TO_STERADIAN

def resample_source_to_instrument_grid(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray) -> xr.DataArray:
    """Resamples a source cube onto an instrument grid, calculating flux per pixel."""

    # 1. Get the 1D coordinate axes from the source cube.
    y_coords = source_cube.coords['y'].values
    x_coords = source_cube.coords['x'].values

    # 2. Create 2D coordinate grids. `indexing='ij'` is crucial to match the (y, x)
    #    dimension order of the data array.
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    # 3. Flatten the 2D coordinate grids. Now `x_source_flat` and `y_source_flat`
    #    will have one entry for every pixel in the source image, matching the
    #    flattened values array.
    x_source_flat = xx.ravel()
    y_source_flat = yy.ravel()

    dx_source = abs(x_coords[1] - x_coords[0])
    dy_source = abs(y_coords[1] - y_coords[0])
    source_pixel_solid_angle = (dx_source * dy_source) * ARCSEC_2_TO_STERADIAN

    source_flux_map = source_cube * source_pixel_solid_angle

    binned_flux_list = []
    for wavelength_slice in source_flux_map:
        statistic, _, _, _ = binned_statistic_2d(
            x=x_source_flat,
            y=y_source_flat,
            values=wavelength_slice.values.ravel(), # Has N*M elements
            statistic='sum',
            bins=[x_edges, y_edges]
        )
        binned_flux_list.append(statistic.T)

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