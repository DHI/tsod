.. _design:

Design philosophy
=================

* Easy to use
* Easy to install
* Easy to get started
* Open Source​
* Easy to collaborate​
* Reproducible
* Easy access to new features


Easy to use
-----------
Common operations such as reading a file should only need a few lines of code.

Make extensive use of existing standard libraries for scientific computing such as numpy, matplotlib and pandas.


Easy to install
---------------

From PyPI::

    pip install tsod


Easy to get started
-------------------
By providing many examples to cut/paste from.

Examples are available in two forms:

* `Unit tests <https://github.com/DHI/tsod/tree/master/tests>`_
* `Jupyter notebooks <https://nbviewer.jupyter.org/github/DHI/tsod/tree/master/notebooks/>`_

Open Source​
------------

tsod is an open source project licensed under the `MIT license <https://github.com/DHI/tsod/blob/master/License>`_.
The software is provided free of charge with the source code available for inspection and modification.


Easy to collaborate​
--------------------

By developing `tsod` on GitHub along with a completely open discussion, we believe that the collaboration between developers and end-users results in a useful library.

Reproducible
------------

By providing the historical versions of `tsod`` on PyPI it is possible to reproduce the behaviour of an older existing system, based on an older version.

Install specific version::

    pip install tsod==0.1.2


Install development version::

    pip install https://github.com/DHI/tsod/archive/main.zip
