import pytest
from unittest.mock import patch
import xarray as xr

# Import the refactored components
from toolkit.modeling.spatial_resampling import resample_source_to_instrument_grid
from toolkit.modeling.resamplers import CPUResampler
from toolkit.defines.modelingsettings import ModelingSettings

@pytest.mark.parametrize(
    "use_gpu, cuda_avail, mlx_avail, expected_class_path",
    [
        # --- The patch paths below have been corrected ---

        # GPU disabled, should always use CPU
        (False, True, True, 'toolkit.modeling.spatial_resampling.CPUResampler'),
        (False, False, False, 'toolkit.modeling.spatial_resampling.CPUResampler'),
        
        # GPU enabled, check selection logic
        (True, False, False, 'toolkit.modeling.spatial_resampling.CPUResampler'), # Fallback
        (True, True, False, 'toolkit.modeling.spatial_resampling.CUDAResampler'), # CUDA preferred
        (True, False, True, 'toolkit.modeling.spatial_resampling.MLXResampler'),  # MLX fallback
        (True, True, True, 'toolkit.modeling.spatial_resampling.CUDAResampler'),   # CUDA preferred over MLX
    ]
)
def test_dispatcher_selects_correct_resampler(
    use_gpu, cuda_avail, mlx_avail, expected_class_path,
    source_cube, nirspec_settings
):
    """
    Tests that the dispatcher function selects the correct resampler class
    based on hardware availability and user settings.
    """
    # Create a new modeling settings instance for this test
    settings = ModelingSettings(
        instrument=nirspec_settings.instrument,
        exposure_time=nirspec_settings.exposure_time,
        use_gpu=use_gpu
    )
    x_edges, y_edges = settings.instrument.get_pixel_layout()

    # Mock the hardware detection and the __init__ of the resampler classes
    with patch('toolkit.modeling.spatial_resampling._is_cuda_available', return_value=cuda_avail), \
         patch('toolkit.modeling.spatial_resampling._is_mlx_available', return_value=mlx_avail), \
         patch(expected_class_path) as MockResampler:

        # Call the dispatcher function
        resample_source_to_instrument_grid(source_cube, x_edges, y_edges, settings)

        # Assert that the expected resampler class was instantiated
        MockResampler.assert_called_once()

def test_cpu_resampler_direct_instantiation(source_cube, nirspec_settings):
    """
    Tests the CPUResampler class directly, ensuring it can be instantiated
    and run in isolation.
    """
    x_edges, y_edges = nirspec_settings.instrument.get_pixel_layout()
    
    # Instantiate the resampler directly
    cpu_resampler = CPUResampler(source_cube, x_edges, y_edges, nirspec_settings)
    
    # Run the resampling
    resampled_cube = cpu_resampler.resample()
    
    # Verify the output
    assert isinstance(resampled_cube, xr.DataArray)
    assert resampled_cube.dims == ('wavelength', 'pix_y', 'pix_x')
    assert resampled_cube.shape[1:] == (30, 30) # NIRSpec is 30x30
    assert not resampled_cube.isnull().any() # Check for NaNs