import os
import pytest

try:
    import vcr
except Exception:  # pragma: no cover - optional
    vcr = None


@pytest.fixture(scope='module')
def vcr_cassette_dir():
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'recordings')


@pytest.fixture
def vcr_fixture(vcr_cassette_dir):
    if vcr is None:
        pytest.skip("vcrpy not installed; skip integration tests that require recording")

    my_vcr = vcr.VCR(serializer='yaml', cassette_library_dir=vcr_cassette_dir, record_mode='once')
    return my_vcr
