import numpy as np
import xarray as xr
from astropy.convolution import AiryDisk2DKernel, convolve
from scipy.stats import binned_statistic_2d

from .base import ResamplerBase

class CPUResampler(ResamplerBase):
    """Vectorized CPU implementation using xarray.apply_ufunc."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("--- No GPU detected or GPU usage disabled. Using vectorized CPU resampling. ---")

    def _apply_psf(self, image_slice: np.ndarray, wavelength_m: float) -> np.ndarray:
        first_null_rad = 1.22 * wavelength_m / self.instrument.mirror_diameter
        first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
        first_null_pix = first_null_arcsec / self.dx_source
        if first_null_pix > 0.1:
            psf_kernel = AiryDisk2DKernel(first_null_pix)
            return convolve(array=image_slice, kernel=psf_kernel, boundary='extend', normalize_kernel=True)
        return image_slice

    def _convolve_and_rebin_slice(self, image_slice: np.ndarray, wavelength_nm: float) -> np.ndarray:
        convolved_slice = self._apply_psf(image_slice, wavelength_nm * 1e-9)
        statistic, _, _, _ = binned_statistic_2d(
            x=self.x_source_flat,
            y=self.y_source_flat,
            values=convolved_slice.ravel(),
            statistic='sum',
            bins=[self.x_edges, self.y_edges]
        )
        return statistic.T

    def resample(self) -> xr.DataArray:
        flux_cube = xr.apply_ufunc(
            self._convolve_and_rebin_slice,
            self.source_flux_map,
            self.source_flux_map.wavelength,
            input_core_dims=[['y', 'x'], []],
            output_core_dims=[['pix_y', 'pix_x']],
            exclude_dims=set(('y', 'x')),
            vectorize=True,
        )
        return self._create_final_cube(flux_cube.data)