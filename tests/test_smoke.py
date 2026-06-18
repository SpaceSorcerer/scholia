import scholia


def test_package_exposes_version():
    assert isinstance(scholia.__version__, str)
    assert scholia.__version__ == "0.1.0"
