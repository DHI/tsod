import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anomalydetection",
    version='0.0.1',
    install_requires=["pandas"],
    extras_require={
        "dev": ["pytest"],
        "test": ["pytest"],
    },
    author="Rasmus Halvgaard",
    author_email="rha@dhigroup.com",
    description="Time series anomaly detection.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/mikeio",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
