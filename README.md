# anomalydetection: Anomaly Detection for time series data.

Sensors often provide faulty or missing observations, i.e. anomalies. Detecting anomalies in time series is a recurring problem in many applications. It is often the first step in data preprocessing and automated data quality assurance that also involves: gap-filling and modelling missing observations in real data sets.

Both in realtime but also for using timeseries data without large errors to run numerical models and decision support tools.

# Goals
This package aims to be:
- Simple to use, install and deploy operationally
- Accessible to everyone (open-source)
- Low computational cost
- Prodive a catalog examples tailored to DHI users and the water domain


# Definitons
Note that we distinguish between [two types of anomaly detection]: https://scikit-learn.org/stable/modules/outlier_detection.html

- Outlier detection (unsupervised anomaly detection)
The training data may contain outliers, i.e. observations far from most other observations. Outlier detectors try to conecntrate on the observations in the training data that similar and close togehter, and ignores observations further away.

- Novelty detection (semi-supervised anomaly detection)
The training data is considered "normal" and is not polluted by outliers. New test data observations can be categorized as an outlier and is in this context called a novelty.


# Installation
pip install git+https://github.com/DHI/anomalydetection.git
