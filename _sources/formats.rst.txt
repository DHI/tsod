.. _formats:

Data formats
###############

Converting data to a :py:class:`~pandas.Series`


    >>> import pandas as pd
    >>> df = pd.read_csv("mydata.csv", parse_dates=True, index_col=0)
    >>> my_series = df['water_level']
