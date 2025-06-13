from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.main.simulatorbase import SimulatorBase
from toolkit.modeling.model_spectrum import calculate_1D_observed_spectrum, convert_specific_intensity_to_flux
from toolkit.read_data.source import read_source_3D_cube 
import xarray as xr

class PointSourceSimulator(SimulatorBase):
    r''' A decorator to observe a source as an unresolved point source.
    
    This simulator reads a 3D data cube but integrates over all spatial
    pixels to create a single 1D spectrum, simulating an instrument that
    cannot resolve the source.

    Example usage:
        filepath = r'path/to/source_datacube.nc'
        sim = PointSourceSimulator(filepath=filepath)
    '''
    def __init__(self, filepath: str):
        '''
        Initializes the simulator and loads the input 3D data cube.
        '''
        super().__init__()
        self.input_spectrum = read_source_3D_cube(filepath=filepath)
        
    def _integrate_fov(self) -> xr.DataArray:
        '''
        Integrates the specific intensity over the spatial dimensions (the FOV)
        to produce a single 1D spectrum for an unresolved source.
        '''
        # Sum the specific intensity over the 'x' and 'y' dimensions.
        # The result is a 1D DataArray with dimensions ('wavelength')
        # The units remain J/s/m^2/nm/sr because we are summing the intensity
        # contributions from each direction (pixel).
        integrated_intensity = self.input_spectrum.sum(dim=['x', 'y'])
        return integrated_intensity

    def get_flux_at_detector(self, modeling_settings: ModelingSettings):
        ''' Return an xarray with the spectrum at the instrument before the addition of noise sources.'''
        
        # First, create the 1D point source spectrum by integrating the cube
        point_source_intensity = self._integrate_fov()

        flux_at_detector = convert_specific_intensity_to_flux(
            input_intensity=point_source_intensity.values, 
            modeling_settings=modeling_settings
        )
        return flux_at_detector

    def get_observed_spectrum(self, modeling_settings: ModelingSettings):
        '''Return an xarray with the observed spectrum, including error bars based on noise sources.'''

        # First, create the 1D point source spectrum by integrating the cube
        point_source_intensity = self._integrate_fov()
        
        # Now, pass this 1D spectrum to the existing calculation function
        observed_spectrum = calculate_1D_observed_spectrum(
            input_spectrum=point_source_intensity, 
            modeling_settings=modeling_settings
        )
        return observed_spectrum
    
    def get_transit_spectrum(self, modeling_settings: ModelingSettings):
        raise NotImplementedError

class IFUSimulator(SimulatorBase):
    ''' A decorator around a 3D datacube from an integral field unit.'''
    def __init__(self, filepath: str):
        '''
        Initializes the simulator and loads the input spectrum data.
        '''
        super().__init__()
        self.input_spectrum = read_source_3D_cube(filepath=filepath)

    def get_flux_at_detector(self, modeling_settings: ModelingSettings):
        raise NotImplementedError

    def get_observed_spectrum(self, modeling_settings: ModelingSettings):
        raise NotImplementedError