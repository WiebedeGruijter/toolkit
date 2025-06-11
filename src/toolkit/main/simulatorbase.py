from abc import ABC, abstractmethod
from toolkit.defines.modelingsettings import ModelingSettings # Assuming this exists

class SimulatorBase(ABC):
    '''
    Abstract Base Class for all simulators.
    
    It defines a common interface for different types of instrument simulators.
    '''
    def __init__(self):
        self.input_spectrum = None

    @abstractmethod
    def get_observed_spectrum(self, modeling_settings: ModelingSettings):
        '''Return an xarray with the observed spectrum, including noise.'''
        raise NotImplementedError

    @abstractmethod
    def get_flux_at_detector(self, modeling_settings: ModelingSettings):
        '''Return an xarray with the spectrum at the instrument before noise.'''
        raise NotImplementedError