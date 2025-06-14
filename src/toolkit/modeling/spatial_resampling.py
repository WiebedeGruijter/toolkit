import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.convolution import AiryDisk2DKernel, convolve
from toolkit.utils.unit_conversions import ARCSEC_2_TO_STERADIAN
from toolkit.defines.modelingsettings import ModelingSettings
import platform

# --- Conditional Imports & Backend Selection ---

BACKEND = 'cpu' # Default to CPU
try:
    import cupy
    import cupyx.scipy.ndimage as cnd
    import cupy_xarray
    BACKEND = 'cuda'
except ImportError:
    # If CUDA is not available, check for Apple Silicon Metal
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import mlx.core as mx
            import mlx.nn as mnn
            BACKEND = 'mlx'
        except ImportError:
            pass

# --- GPU (CUDA) Functions ---

if BACKEND == 'cuda':
    def _apply_psf_cuda(image_slice: cupy.ndarray, wavelength_m: float, mirror_diameter: float, pixel_scale_arcsec: float) -> cupy.ndarray:
        # (Code is identical to before)
        first_null_rad = 1.22 * wavelength_m / mirror_diameter
        first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
        first_null_pix = first_null_arcsec / pixel_scale_arcsec
        if first_null_pix > 0.1:
            psf_kernel_cpu = AiryDisk2DKernel(first_null_pix)
            psf_kernel_gpu = cupy.asarray(psf_kernel_cpu)
            convolved_slice = cnd.convolve(input=image_slice, weights=psf_kernel_gpu, mode='constant', cval=0.0)
            return convolved_slice / cupy.sum(psf_kernel_gpu)
        else:
            return image_slice

    def _convolve_and_rebin_chunk_cuda(image_chunk_gpu: cupy.ndarray, wavelength_chunk_nm: np.ndarray, x_source_flat_gpu: cupy.ndarray, y_source_flat_gpu: cupy.ndarray, x_edges_gpu: cupy.ndarray, y_edges_gpu: cupy.ndarray, mirror_diameter: float, pixel_scale_arcsec: float) -> cupy.ndarray:
        # (Code is identical to before)
        binned_slices = []
        for i in range(image_chunk_gpu.shape[0]):
            image_slice_gpu = image_chunk_gpu[i, :, :]
            wavelength_nm = wavelength_chunk_nm[i]
            convolved_slice_gpu = _apply_psf_cuda(image_slice=image_slice_gpu, wavelength_m=wavelength_nm * 1e-9, mirror_diameter=mirror_diameter, pixel_scale_arcsec=pixel_scale_arcsec)
            statistic_gpu, _, _ = cupy.histogram2d(x=x_source_flat_gpu, y=y_source_flat_gpu, bins=[x_edges_gpu, y_edges_gpu], weights=convolved_slice_gpu.ravel())
            binned_slices.append(statistic_gpu.T)
        return cupy.array(binned_slices)

    def _resample_cuda(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings) -> xr.DataArray:
        # (Code is identical to before)
        print("--- NVIDIA CUDA GPU detected. Running fully accelerated resampling. ---")
        source_cube_gpu = source_cube.cupy.as_cupy()
        dx_source = abs(source_cube.coords['x'].values[1] - source_cube.coords['x'].values[0])
        source_pixel_solid_angle = (dx_source**2) * ARCSEC_2_TO_STERADIAN
        source_flux_map_gpu = source_cube_gpu * source_pixel_solid_angle
        wavelength_values_cpu = source_cube.coords['wavelength'].values
        
        y_coords, x_coords = source_cube.coords['y'].values, source_cube.coords['x'].values
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        x_source_flat_gpu, y_source_flat_gpu = cupy.asarray(xx.ravel()), cupy.asarray(yy.ravel())
        x_edges_gpu, y_edges_gpu = cupy.asarray(x_edges), cupy.asarray(y_edges)

        output_sizes = {"pix_y": len(y_edges) - 1, "pix_x": len(x_edges) - 1}

        flux_cube_dask = xr.apply_ufunc(_convolve_and_rebin_chunk_cuda, source_flux_map_gpu, wavelength_values_cpu, input_core_dims=[["wavelength", "y", "x"], ["wavelength"]], output_core_dims=[["wavelength", "pix_y", "pix_x"]], exclude_dims=set(("y", "x")), dask="parallelized", output_dtypes=[source_flux_map_gpu.dtype], output_sizes=output_sizes, kwargs={"x_source_flat_gpu": x_source_flat_gpu, "y_source_flat_gpu": y_source_flat_gpu, "x_edges_gpu": x_edges_gpu, "y_edges_gpu": y_edges_gpu, "mirror_diameter": modeling_settings.instrument.mirror_diameter, "pixel_scale_arcsec": dx_source})
        
        final_cube = flux_cube_dask.compute()
        final_cube = final_cube.assign_coords(wavelength=source_cube.coords['wavelength'], pix_y=np.arange(len(y_edges) - 1), pix_x=np.arange(len(x_edges) - 1))
        final_cube.attrs["units"] = "W m^-2 nm^-1"
        return final_cube

# --- GPU (Apple Metal) Functions ---

if BACKEND == 'mlx':
    def _apply_psf_mlx(image_slice: np.ndarray, wavelength_m: float, mirror_diameter: float, pixel_scale_arcsec: float) -> np.ndarray:
        """Performs convolution on Apple Silicon GPU, returns result to CPU."""
        first_null_rad = 1.22 * wavelength_m / mirror_diameter
        first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
        first_null_pix = first_null_arcsec / pixel_scale_arcsec
        if first_null_pix > 0.1:
            # 1. Prepare arrays for MLX convolution
            psf_kernel_np = AiryDisk2DKernel(first_null_pix).array
            image_mlx = mx.array(image_slice[np.newaxis, ..., np.newaxis], dtype=mx.float32) # Add Batch and Channel dims
            kernel_mlx = mx.array(psf_kernel_np[..., np.newaxis, np.newaxis], dtype=mx.float32) # Add In/Out Channel dims

            # 2. Perform 2D convolution on Metal GPU
            convolved_mlx = mnn.convolution.conv2d(image_mlx, kernel_mlx, stride=1, padding=psf_kernel_np.shape[0] // 2)
            
            # 3. Trigger computation and return to NumPy array on CPU
            return np.array(convolved_mlx[0, :, :, 0])
        else:
            return image_slice
            
    def _resample_mlx(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings) -> xr.DataArray:
        """A hybrid approach: convolution on MLX, binning on CPU."""
        print("--- Apple Silicon GPU detected. Running partially accelerated resampling. ---")
        y_coords, x_coords = source_cube.coords['y'].values, source_cube.coords['x'].values
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        x_source_flat, y_source_flat = xx.ravel(), yy.ravel()
        
        dx_source = abs(x_coords[1] - x_coords[0])
        source_pixel_solid_angle = (dx_source**2) * ARCSEC_2_TO_STERADIAN
        source_flux_map = source_cube * source_pixel_solid_angle

        binned_flux_list = []
        for wl in source_flux_map.wavelength.values:
            wl_slice = source_flux_map.sel(wavelength=wl)
            # Use the MLX-accelerated convolution function
            convolved_slice = _apply_psf_mlx(image_slice=wl_slice.values, wavelength_m=wl.item() * 1e-9, mirror_diameter=modeling_settings.instrument.mirror_diameter, pixel_scale_arcsec=dx_source)
            # Use the standard CPU binning function
            statistic, _, _, _ = binned_statistic_2d(x=x_source_flat, y=y_source_flat, values=convolved_slice.ravel(), statistic='sum', bins=[x_edges, y_edges])
            binned_flux_list.append(statistic.T)

        flux_cube = xr.DataArray(data=np.array(binned_flux_list), dims=['wavelength', 'pix_y', 'pix_x'], coords={'wavelength': source_cube.coords['wavelength'], 'pix_y': np.arange(len(y_edges) - 1), 'pix_x': np.arange(len(x_edges) - 1)}, attrs={"units": "W m^-2 nm^-1"})
        return flux_cube

# --- CPU-Only Functions ---

def _apply_psf_cpu(image_slice: np.ndarray, wavelength_m: float, mirror_diameter: float, pixel_scale_arcsec: float) -> np.ndarray:
    first_null_rad = 1.22 * wavelength_m / mirror_diameter
    first_null_arcsec = np.rad2deg(first_null_rad) * 3600.0
    first_null_pix = first_null_arcsec / pixel_scale_arcsec
    if first_null_pix > 0.1:
        psf_kernel = AiryDisk2DKernel(first_null_pix)
        return convolve(array=image_slice, kernel=psf_kernel, boundary='extend', normalize_kernel=True)
    else:
        return image_slice

def _convolve_and_rebin_slice(
    image_slice: np.ndarray,
    wavelength_nm: float,
    x_source_flat: np.ndarray,
    y_source_flat: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mirror_diameter: float,
    pixel_scale_arcsec: float
) -> np.ndarray:
    """Core CPU operation for vectorization: Convolves and rebins a single 2D slice."""
    # 1. Apply PSF using the CPU helper function
    convolved_slice = _apply_psf_cpu(
        image_slice=image_slice,
        wavelength_m=wavelength_nm * 1e-9,
        mirror_diameter=mirror_diameter,
        pixel_scale_arcsec=pixel_scale_arcsec
    )
    # 2. Perform binning with SciPy
    statistic, _, _, _ = binned_statistic_2d(
        x=x_source_flat,
        y=y_source_flat,
        values=convolved_slice.ravel(),
        statistic='sum',
        bins=[x_edges, y_edges]
    )
    return statistic.T

def _resample_cpu(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings) -> xr.DataArray:
    """
    Vectorized CPU implementation that uses xr.apply_ufunc to loop over
    the wavelength dimension. Runs on a single core.
    """
    print("--- No GPU detected. Running vectorized CPU resampling. ---")
    y_coords, x_coords = source_cube.coords['y'].values, source_cube.coords['x'].values
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    x_source_flat, y_source_flat = xx.ravel(), yy.ravel()
    
    dx_source = abs(x_coords[1] - x_coords[0])
    source_pixel_solid_angle = (dx_source**2) * ARCSEC_2_TO_STERADIAN
    source_flux_map = source_cube * source_pixel_solid_angle

    # Use apply_ufunc with vectorize=True.
    # This tells xarray to intelligently broadcast our function over the non-core dimensions.
    flux_cube = xr.apply_ufunc(
        _convolve_and_rebin_slice,      # Our simple, single-slice function
        source_flux_map,                # The 3D input cube
        source_flux_map.wavelength,     # The 1D wavelength coordinate array
        input_core_dims=[['y', 'x'], []], # Defines a 2D slice and a scalar as inputs
        output_core_dims=[['pix_y', 'pix_x']], # Defines a 2D slice as the output
        exclude_dims=set(('y', 'x')),   # Dimensions to drop from the output
        vectorize=True,                 # The key change!
        kwargs={                        # Constant arguments for every call
            "x_source_flat": x_source_flat,
            "y_source_flat": y_source_flat,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "mirror_diameter": modeling_settings.instrument.mirror_diameter,
            "pixel_scale_arcsec": dx_source,
        }
    )

    flux_cube = flux_cube.assign_coords(
        pix_y=np.arange(len(y_edges) - 1),
        pix_x=np.arange(len(x_edges) - 1)
    )
    flux_cube.attrs["units"] = "W m^-2 nm^-1"
    return flux_cube

# --- Main Public Function (The Dispatcher) ---

def resample_source_to_instrument_grid(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings) -> xr.DataArray:
    """
    Resamples a source cube onto an instrument grid. This function intelligently
    selects the fastest available hardware backend (NVIDIA CUDA, Apple Metal, or CPU).
    """
    # Added a simple flag to the dispatcher to control GPU usage
    if BACKEND == 'cuda' and modeling_settings.use_gpu:
        source_cube_dask = source_cube.chunk("auto")
        gpu_result = _resample_cuda(source_cube_dask, x_edges, y_edges, modeling_settings)
        return gpu_result.cupy.as_numpy()

    elif BACKEND == 'mlx' and modeling_settings.use_gpu:
        return _resample_mlx(source_cube, x_edges, y_edges, modeling_settings)
        
    else: # Fallback to CPU
        return _resample_cpu(source_cube, x_edges, y_edges, modeling_settings)