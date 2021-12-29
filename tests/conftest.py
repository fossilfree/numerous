import pytest


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield
