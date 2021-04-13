import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsod",
    version="0.1.3",
    install_requires=["pandas>=1.0.0", "numba", "joblib"],
    extras_require={
        "dev": ["pytest>=6.2.1", "sphinx", "sphinx-rtd-theme"],
        "ml": ["pyod", "tensorflow"],
        "test": ["pytest>=6.2.1"],
    },
    author="Henrik Andersson",
    author_email="jan@dhigroup.com",
    description="Time series anomaly detection.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/tsod",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
