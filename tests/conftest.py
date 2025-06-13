import pytest
from pathlib import Path
import numpy as np

# Import classes from your toolkit
from toolkit.defines.instrument import JWSTMiri, JWSTNirSpecIFU
from toolkit.defines.modelingsettings import ModelingSettings
from toolkit.main.simulator import InstrumentSimulator

TEST_DATA_DIR = Path(__file__).parent / 'test_data'

@pytest.fixture
def test_data_path() -> Path:
    """Fixture to provide the path to the test_data directory."""
    return TEST_DATA_DIR

@pytest.fixture
def datacube_filepath(test_data_path: Path) -> str:
    """Fixture to provide the full path to the test datacube."""
    return str(test_data_path / 'datacube.nc')

@pytest.fixture
def point_source_filepaths(test_data_path: Path) -> dict[str, Path]:
    """Fixture to provide paths to the point source test files."""
    return {
        'T2000': test_data_path / 'blackbody_T2000.txt',
        'T5000': test_data_path / 'blackbody_T5000.txt',
    }

@pytest.fixture
def miri_instrument() -> JWSTMiri:
    """Fixture to create an instance of the JWSTMiri instrument."""
    return JWSTMiri()

@pytest.fixture
def nirspec_instrument() -> JWSTNirSpecIFU:
    """Fixture to create an instance of the JWSTNirSpecIFU instrument."""
    return JWSTNirSpecIFU()

@pytest.fixture
def miri_settings(miri_instrument: JWSTMiri) -> ModelingSettings:
    """Fixture for MIRI modeling settings with a defined exposure time."""
    return ModelingSettings(instrument=miri_instrument, exposure_time=1000.0)

@pytest.fixture
def nirspec_settings(nirspec_instrument: JWSTNirSpecIFU) -> ModelingSettings:
    """Fixture for NIRSpec modeling settings with a defined exposure time."""
    return ModelingSettings(instrument=nirspec_instrument, exposure_time=1000.0)

@pytest.fixture
def instrument_simulator(datacube_filepath: str) -> InstrumentSimulator:
    """Fixture to create an InstrumentSimulator with the test datacube."""
    return InstrumentSimulator(filepath=datacube_filepath)