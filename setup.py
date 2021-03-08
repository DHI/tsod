import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anomalydetection",
    version='0.1.0',
    install_requires=["pandas>=1.2.0", "numba"],
    extras_require={
        "dev": ["pytest>=6.2.1"],
        "ml": ["pyod", "keras", "tensorflow"],
        "test": ["pytest>=6.2.1"],
    },
    author="Rasmus Halvgaard",
    author_email="rha@dhigroup.com",
    description="Time series anomaly detection.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/anomalydetection",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
