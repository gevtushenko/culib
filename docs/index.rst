.. CULIB documentation master file, created by
   sphinx-quickstart on Tue Apr  7 23:59:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========
**CULIB** is an open-source library providing fast and flexible software for every layer of
the CUDA programming model. The general goal of this project is to extend algorithms beyond
warp, block, and device levels. **CULIB** also implements multi-GPU and multi-node versions of its
algorithms.

**CULIB** is GPU oriented. It means that a result of any algorithm is stored on GPU and never copied on CPU. This
also means that any **CULIB** components can be used only within kernel or device functions.

**CULIB** is not oriented on rapid prototyping of CUDA applications. Instead, **CULIB** is designed to raise the abstraction
of complex algorithms without performance overhead. **CULIB** relies on policy-based design to let you tune the algorithms
for specific cases.

Examples
========

**CULIB** provides algorithms as templated classes. Objects of these classes could be reused. The following code
illustrates a kernel fragment performing :ref:`exclusive scan <exclusive>` across the threads of a warp:

.. code-block:: cuda

   using namespace culib;
   warp::scan<int> scan;

   const int sum_result = scan.exclusive (flag);
   const int max_result = scan.exclusive (in[sum_result], binary_op::max<data_type> {});


.. Index

.. toctree::
   :hidden:

   warp
