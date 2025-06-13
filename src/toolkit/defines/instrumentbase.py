from abc import ABC, abstractmethod
import numpy as np

class InstrumentBase(ABC):
    """
    Abstract base class for any instrument.
    
    It enforces the implementation of the get_pixel_layout method, which is
    fundamental for projecting a source onto the instrument's detector.
    """
    @abstractmethod
    def get_pixel_layout(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the angular coordinates of the detector pixel edges.
        This defines the grid onto which the source sky is projected.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the array of
                                           pixel edges along the x-axis and y-axis,
                                           in arcseconds.
        """
        pass

class JWST(InstrumentBase):
    """
    A base class for all JWST instruments, inheriting the abstract methods
    from InstrumentBase.
    """
    mirror_diameter = 2.4