from toolkit.defines.instrumentbase import JWST
import numpy as np

class JWSTMiri(JWST):
    """
    Models the MIRI MRS instrument. For the purpose of spatially-integrated
    spectroscopy, we can treat its FOV as a single large 'pixel'.
    """
    def get_pixel_layout(self) -> tuple[np.ndarray, np.ndarray]:
        # Based on https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-medium-resolution-spectroscopy
        fov_x_arcsec = 11.0
        fov_y_arcsec = 12.0
        
        # A 1x1 pixel grid is defined by two edges on each axis.
        x_edges = np.array([-fov_x_arcsec / 2, fov_x_arcsec / 2])
        y_edges = np.array([-fov_y_arcsec / 2, fov_y_arcsec / 2])
        return x_edges, y_edges

class JWSTNirSpecIFU(JWST):
    """
    Models the NIRSpec IFU, which has a 3"x3" FOV sampled by a 30x30 grid
    of pixels (spaxels).
    """
    def get_pixel_layout(self) -> tuple[np.ndarray, np.ndarray]:
        # Based on https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-modes/nirspec-integral-field-unit-spectroscopy
        pixel_scale_arcsec = 0.1
        num_pixels = 30
        fov_arcsec = pixel_scale_arcsec * num_pixels # Total FOV is 3.0"

        # To define 30 pixels, we need 31 edges. Center the grid at (0,0).
        edges = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, num_pixels + 1)
        
        x_edges = edges
        y_edges = edges
        return x_edges, y_edges