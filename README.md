# tsod: Anomaly Detection for time series data.
[![Full test](https://github.com/DHI/tsod/actions/workflows/python-app.yml/badge.svg)](https://github.com/DHI/tsod/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/tsod.svg)](https://badge.fury.io/py/tsod)
![Python version](https://img.shields.io/pypi/pyversions/tsod.svg) 

![univariate](https://raw.githubusercontent.com/DHI/tsod/main/images/anomaly.png)

Sensors often provide faulty or missing observations. These anomalies must be detected automatically and replaced with more feasible values before feeding the data to numerical simulation engines as boundary conditions or real time decision systems.

This package aims to provide examples and algorithms for detecting anomalies in time series data specifically tailored to DHI users and the water domain. It is simple to install and deploy operationally and is accessible to everyone (open-source).

# Getting Started

* [Documentation](https://dhi.github.io/tsod/getting_started.html)
* [Notebook](https://github.com/DHI/tsod/blob/main/notebooks/Getting%20started.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/DHI/tsod/blob/main/notebooks/Getting%20started.ipynb)


# Installation

`tsod` is a pure Python library and runs on Windows, Linux and Mac.

From PyPI:

`pip install tsod`

Or development version:

`pip install https://github.com/DHI/tsod/archive/main.zip`

# Vision
* A simple and consistent API for anomaly detection of timeseries
* The computational speed will be good for typical timeseries data found in the water domain, to support realtime detection
* It will have a suite of different algorithms ranging from simple rule-based to more advanced based on e.g. neural networks

# Definitions
Note that we distinguish between [two types of anomaly detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

- Outlier detection (unsupervised anomaly detection)
The training data may contain outliers, i.e. observations far from most other observations. Outlier detectors try to concentrate on the observations in the training data that similar and close together, and ignores observations further away.

- Novelty detection (semi-supervised anomaly detection)
The training data is considered "normal" and is not polluted by outliers. New test data observations can be categorized as an outlier and is in this context called a novelty.


# Contribute to `tsod`
[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/DHI/tsod)
- Follow PEP8 code style. This is automatically checked during Pull Requests.

- Raise custom exceptions. This makes it easier to catch and separate built-in errors from our own throws.

- If citing or re-using other code please make sure their license is also consistent with our policy.

