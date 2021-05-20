.. _getting_started:

Getting started
===============

`tsod` is library for timeseries data. The format of a timeseries is always a :py:class:`~pandas.Series` and in some cases with a :py:class:`~pandas.DatetimeIndex` 

1. Get data in the form of a a :py:class:`~pandas.Series` (see Data formats below) 
2. Select one or more detectors e.g. :class:`RangeDetector <tsod.RangeDetector>` or :class:`ConstantValueDetector <tsod.ConstantValueDetector>`
3. Define parameters (e.g. min/max, max rate of change) or...
4. Fit parameters based on normal data, i.e. without outliers
5. Detect outliers in any dataset

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: http://colab.research.google.com/github/DHI/tsod/blob/main/notebooks/Getting%20started.ipynb

Example
-------

>>> import pandas as pd
>>> from tsod import RangeDetector
>>> rd = RangeDetector(max_value=2.0)
>>> data = pd.Series([0.0, 1.0, 3.0]) # 3.0 is out of range i.e. an anomaly
>>> anom = rd.detect(data)
>>> anom
  0    False
  1    False
  2     True
  dtype: bool
>>> data[anom] # get anomalous data
2    3.0
dtype: float64
>>> data[~anom] # get normal data
0    0.0
1    1.0
dtype: float64
>>> 


Saving and loading
------------------
.. code-block:: python

    # save a configured detector
    cd = CombinedDetector([ConstantValueDetector(), RangeDetector()])
    cd.fit(normal_data)
    cd.save("detector.joblib")

    # ... and then later load it from disk
    my_detector = tsod.load("detector.joblib")
    my_detector.detect(some_data)

Data formats
------------
    
Converting data to a :py:class:`~pandas.Series`
    
.. code-block:: python

    import pandas as pd
    df = pd.read_csv("mydata.csv", parse_dates=True, index_col=0)
    my_series = df['water_level']

    from mikeio import Dfs0
    dfs = Dfs0('simple.dfs0')
    df = dfs.to_dataframe()
    my_series_2 = df['rainfall']
    