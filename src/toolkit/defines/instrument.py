from toolkit.defines.instrumentbase import JWST

class JWSTMiri(JWST):
    fov_x_arcsec: float = 11.0
    fov_y_arcsec: float = 12.0

class JWSTNirSpecIFU(JWST):
    pass