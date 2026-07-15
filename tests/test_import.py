"""Minimal smoke tests: the package imports and exposes its public API."""


def test_import():
    import nfi  # noqa: F401


def test_public_api():
    from nfi import nFIEstimator  # noqa: F401


def test_version():
    import nfi
    assert isinstance(nfi.__version__, str)
    assert nfi.__version__  # non-empty