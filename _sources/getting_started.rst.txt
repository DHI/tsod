.. _getting_started:

Getting started
###############

`tsod` is library for timeseries data. The format of a timeseries is always a :py:class:`~pandas.Series` with a :py:class:`~pandas.DatetimeIndex` 

1. Get data in the form of a a :py:class:`~pandas.Series` with a :py:class:`~pandas.DatetimeIndex` (example :ref:`formats`) 
2. Select one or more detectors e.g. :class:`RangeDetector <tsod.RangeDetector>` or :class:`ConstantValueDetector <tsod.ConstantValueDetector>`
3. Define parameters (e.g. min/max, max rate of change) or...
4. Fit parameters based on normal data, i.e. without outliers
5. Detect outliers in any dataset

.. toctree::
    :maxdepth: 1
    :hidden:
 
    formats