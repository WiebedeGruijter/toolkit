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
        # The core MLX library contains the stateless, functional API
        import mlx.core as mx
        self.mx = mx

    def _apply_psf_mlx(self, image_slice: np.ndarray, wavelength_m: float) -> np.ndarray:
        """Applies the wavelength-dependent Point Spread Function (PSF) using MLX for convolution."""
        # Calculate PSF size based on wavelength and telescope diameter
        first_null_rad = 1.22 * wavelength_m / self.instrument.mirror_diameter
        first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
        first_null_pix = first_null_arcsec / self.dx_source

        # If the PSF is smaller than a pixel, convolution is unnecessary
        if first_null_pix <= 0.1:
            return image_slice

        # 1. Prepare inputs for MLX convolution
        psf_kernel_np = AiryDisk2DKernel(first_null_pix).array

        # mlx.core.conv2d expects input image in NHWC format: (Batch, Height, Width, Channels)
        image_mlx = self.mx.array(image_slice[np.newaxis, ..., np.newaxis], dtype=self.mx.float32)

        # mlx.core.conv2d expects kernel in (out_channels, H, W, in_channels) format
        k_h, k_w = psf_kernel_np.shape
        kernel_mlx = self.mx.array(psf_kernel_np.reshape(1, k_h, k_w, 1), dtype=self.mx.float32)

        # 2. Perform convolution using the correct functional API: mx.conv2d
        padding = k_h // 2
        
        convolved_mlx = self.mx.conv2d(
            image_mlx,
            kernel_mlx,
            stride=1,
            padding=padding
        )

        # Normalize the result to conserve flux, consistent with other resamplers
        convolved_mlx = convolved_mlx / self.mx.sum(kernel_mlx)

        # 3. Convert back to a NumPy array, removing the batch and channel dimensions
        # mx.eval() ensures the lazy computation is finished before returning.
        convolved_np = np.array(convolved_mlx[0, :, :, 0])
        self.mx.eval(convolved_np)

        return convolved_np

    def _convolve_and_rebin_slice(self, image_slice: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:
        """
        Core function for apply_ufunc. It convolves a single 2D slice with the PSF
        and then rebins it to the detector's pixel grid.
        """
        # Convert wavelength to meters for the PSF calculation
        convolved_slice = self._apply_psf_mlx(image_slice, wavelength_nm.item() * 1e-9)

        # Rebin the convolved slice using CPU-based binned_statistic_2d
        statistic, _, _, _ = binned_statistic_2d(
            x=self.x_source_flat,
            y=self.y_source_flat,
            values=convolved_slice.ravel(),
            statistic='sum',
            bins=[self.x_edges, self.y_edges]
        )
        # binned_statistic_2d returns (x_bins, y_bins), but xarray expects (y_bins, x_bins)
        return statistic.T

    def resample(self) -> xr.DataArray:
        """
        Resamples the source cube using a vectorized approach. It applies the
        MLX-accelerated convolution and CPU-based rebinning across all wavelength slices.
        """
        flux_cube = xr.apply_ufunc(
            self._convolve_and_rebin_slice,
            self.source_flux_map,
            self.source_flux_map.wavelength,
            input_core_dims=[['y', 'x'], []],       # Process one y-x slice and one wavelength value at a time
            output_core_dims=[['pix_y', 'pix_x']], # The output for each slice is a detector-sized image
            exclude_dims=set(('y', 'x')),          # Remove original spatial dimensions from the final output
            vectorize=True,                        # Efficiently loop over the non-core 'wavelength' dimension
        )

        return self._create_final_cube(flux_cube.data)