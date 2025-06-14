import platform
import numpy as np
import xarray as xr

from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.modeling.resamplers import CPUResampler, CUDAResampler, MLXResampler

# --- Hardware Detection ---

def _is_cuda_available():
    try:
        import cupy
        return cupy.is_available()
    except ImportError:
        return False

def _is_mlx_available():
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import mlx.core
            mlx.core.zeros(1) # Try a basic operation
            return True
        except (ImportError, RuntimeError):
            return False
    return False

# --- Main Public Function (The Dispatcher) ---

def resample_source_to_instrument_grid(source_cube: xr.DataArray, x_edges: np.ndarray, y_edges: np.ndarray, modeling_settings: ModelingSettings) -> xr.DataArray:
    """
    Resamples a source cube onto an instrument grid. This function intelligently
    selects the fastest available hardware backend (NVIDIA CUDA, Apple Metal,
    or CPU) based on user settings and installed libraries.

    Args:
        source_cube (xr.DataArray): The input 3D data cube (wavelength, y, x).
        x_edges (np.ndarray): The detector pixel edges along the x-axis.
        y_edges (np.ndarray): The detector pixel edges along the y-axis.
        modeling_settings (ModelingSettings): The settings for the simulation.

    Returns:
        xr.DataArray: The resampled data cube on the instrument grid.
    """
    use_gpu = modeling_settings.use_gpu
    resampler_class = CPUResampler # Default

    if use_gpu:
        if _is_cuda_available():
            resampler_class = CUDAResampler
        elif _is_mlx_available():
            resampler_class = MLXResampler
            
    # Instantiate the selected resampler class
    resampler = resampler_class(source_cube, x_edges, y_edges, modeling_settings)

    # Execute the resampling
    return resampler.resample()