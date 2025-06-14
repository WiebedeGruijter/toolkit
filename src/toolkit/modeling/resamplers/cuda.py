import numpy as np
import xarray as xr
from astropy.convolution import AiryDisk2DKernel

from .base import ResamplerBase

class CUDAResampler(ResamplerBase):
    """NVIDIA CUDA-accelerated implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("--- NVIDIA CUDA GPU detected. Running fully accelerated resampling. ---")
        # Local imports to avoid errors if cupy is not installed
        import cupy
        import cupyx.scipy.ndimage as cnd
        import cupy_xarray
        self.cupy = cupy
        self.cnd = cnd

    def _apply_psf_cuda(self, image_slice_gpu: 'cupy.ndarray', wavelength_m: float) -> 'cupy.ndarray':
        first_null_rad = 1.22 * wavelength_m / self.instrument.mirror_diameter
        first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
        first_null_pix = first_null_arcsec / self.dx_source
        if first_null_pix > 0.1:
            psf_kernel_cpu = AiryDisk2DKernel(first_null_pix)
            psf_kernel_gpu = self.cupy.asarray(psf_kernel_cpu)
            convolved_slice = self.cnd.convolve(input=image_slice_gpu, weights=psf_kernel_gpu, mode='constant', cval=0.0)
            return convolved_slice / self.cupy.sum(psf_kernel_gpu)
        return image_slice_gpu

    def _convolve_and_rebin_chunk_cuda(self, image_chunk_gpu, wavelength_chunk_nm, x_source_flat_gpu, y_source_flat_gpu, x_edges_gpu, y_edges_gpu):
        binned_slices = []
        for i in range(image_chunk_gpu.shape[0]):
            convolved_slice_gpu = self._apply_psf_cuda(image_chunk_gpu[i, :, :], wavelength_chunk_nm[i] * 1e-9)
            statistic_gpu, _, _ = self.cupy.histogram2d(x=x_source_flat_gpu, y=y_source_flat_gpu, bins=[x_edges_gpu, y_edges_gpu], weights=convolved_slice_gpu.ravel())
            binned_slices.append(statistic_gpu.T)
        return self.cupy.array(binned_slices)

    def resample(self) -> xr.DataArray:
        source_flux_map_gpu = self.source_flux_map.chunk("auto").cupy.as_cupy()
        x_source_flat_gpu = self.cupy.asarray(self.x_source_flat)
        y_source_flat_gpu = self.cupy.asarray(self.y_source_flat)
        x_edges_gpu, y_edges_gpu = self.cupy.asarray(self.x_edges), self.cupy.asarray(self.y_edges)
        
        output_sizes = {"pix_y": len(self.y_edges) - 1, "pix_x": len(self.x_edges) - 1}

        flux_cube_dask = xr.apply_ufunc(
            self._convolve_and_rebin_chunk_cuda,
            source_flux_map_gpu,
            self.source_cube.coords['wavelength'].values,
            input_core_dims=[["wavelength", "y", "x"], ["wavelength"]],
            output_core_dims=[["wavelength", "pix_y", "pix_x"]],
            exclude_dims=set(("y", "x")),
            dask="parallelized",
            output_dtypes=[source_flux_map_gpu.dtype],
            output_sizes=output_sizes,
            kwargs={
                "x_source_flat_gpu": x_source_flat_gpu,
                "y_source_flat_gpu": y_source_flat_gpu,
                "x_edges_gpu": x_edges_gpu,
                "y_edges_gpu": y_edges_gpu,
            }
        )
        
        final_cube_gpu = flux_cube_dask.compute()
        return self._create_final_cube(final_cube_gpu.cupy.as_numpy())