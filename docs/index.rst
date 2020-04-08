.. CULIB documentation master file, created by
   sphinx-quickstart on Tue Apr  7 23:59:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API Reference
=============

.. toctree::
   :maxdepth: 2

   warp


Core API
========

There are inclusive and exclusive per-warp scan operations in the **CULIB** that are
provided by ``culib::warp::scan`` class. It's important to note that :ref:`exclusive scan <exclusive>`
is implemented as ``culib::warp::scan::inclusive`` combined with values shuffle.
