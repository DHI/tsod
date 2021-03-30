.. _getting_started:

Getting started
===============

`tsod` is library for timeseries data. The format of a timeseries is always a :py:class:`~pandas.Series` and in some cases with a :py:class:`~pandas.DatetimeIndex` 

1. Get data in the form of a a :py:class:`~pandas.Series` (see Data formats below) 
2. Select one or more detectors e.g. :class:`RangeDetector <tsod.RangeDetector>` or :class:`ConstantValueDetector <tsod.ConstantValueDetector>`
3. Define parameters (e.g. min/max, max rate of change) or...
4. Fit parameters based on normal data, i.e. without outliers
5. Detect outliers in any dataset


Saving and loading
------------------

    >>> cd = CombinedDetector([ConstantValueDetector(), RangeDetector()])
    >>> cd.fit(normal_data)
    >>> cd.save("detector.joblib")

    >>> my_detector = tsod.load("detector.joblib")
    >>> my_detector.detect(some_data)

Data formats
------------
    
Converting data to a :py:class:`~pandas.Series`
    
    
    >>> import pandas as pd
    >>> df = pd.read_csv("mydata.csv", parse_dates=True, index_col=0)
    >>> my_series = df['water_level']

    >>> from mikeio import Dfs0
    >>> dfs = Dfs0('simple.dfs0')
    >>> df = dfs.to_dataframe()
    >>> my_series_2 = df['rainfall']
    