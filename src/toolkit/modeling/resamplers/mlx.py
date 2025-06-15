import numpy as np
import xarray as xr
from astropy.convolution import AiryDisk2DKernel
from scipy.stats import binned_statistic_2d

from .base import ResamplerBase

class MLXResampler(ResamplerBase):
    """Apple Silicon (Metal) GPU implementation with MLX for convolution."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("--- Apple Silicon GPU detected. Running partially accelerated resampling (MLX + CPU). ---")
        import mlx.core as mx
        import mlx.nn as mnn
        self.mx = mx
        self.mnn = mnn

    def _apply_psf_mlx(self, image_slice: np.ndarray, wavelength_m: float) -> np.ndarray:
        first_null_rad = 1.22 * wavelength_m / self.instrument.mirror_diameter
        first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
        first_null_pix = first_null_arcsec / self.dx_source
        if first_null_pix > 0.1:
            psf_kernel_np = AiryDisk2DKernel(first_null_pix).array
            image_mlx = self.mx.array(image_slice[np.newaxis, ..., np.newaxis], dtype=self.mx.float32)
            kernel_mlx = self.mx.array(psf_kernel_np[..., np.newaxis, np.newaxis], dtype=self.mx.float32)
            convolved_mlx = self.mnn.convolution.Conv2d(image_mlx, kernel_mlx, stride=1, padding=psf_kernel_np.shape[0] // 2)
            return np.array(convolved_mlx[0, :, :, 0])
        return image_slice

    def resample(self) -> xr.DataArray:
        binned_flux_list = []
        for wl in self.source_flux_map.wavelength.values:
            wl_slice = self.source_flux_map.sel(wavelength=wl)
            convolved_slice = self._apply_psf_mlx(wl_slice.values, wl.item() * 1e-9)
            statistic, _, _, _ = binned_statistic_2d(
                x=self.x_source_flat,
                y=self.y_source_flat,
                values=convolved_slice.ravel(),
                statistic='sum',
                bins=[self.x_edges, self.y_edges]
            )
            binned_flux_list.append(statistic.T)
        
        return self._create_final_cube(np.array(binned_flux_list))