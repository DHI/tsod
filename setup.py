import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsod",
    version="0.2.0",
    install_requires=["pandas>=1.0.0", "numba", "joblib"],
    extras_require={
        "dev": [
            "pytest>=6.2.1",
            "sphinx==4.5.0",  # pin version to work with sphinx-book-theme,
            "sphinx-book-theme",
        ],
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
