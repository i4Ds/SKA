.. Karabo-Pipeline documentation master file, created by
   sphinx-quickstart on Thu Feb 24 13:19:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Karabo-Pipeline's documentation!
===========================================
Karabo is a starting point for the `Square Kilometer Array <https://www.skatelescope.org/the-ska-project/>`_ Digital Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment.

.. toctree::
   :maxdepth: 2
   :caption: Users

   installation_user
   container
   examples/examples.md
   parallel_processing


Modules
===============
.. toctree::
   :maxdepth: 2
   :caption: Simulation

   main_features/simulation.rst

.. toctree::
   :maxdepth: 2
   :caption: Imaging

   main_features/imaging.rst
   main_features/base_imaging.rst
   main_features/oskar_imaging.rst
   main_features/rascil_imaging.rst
   main_features/wsclean_imaging.rst

.. toctree::
   :maxdepth: 2
   :caption: Source Detection

   main_features/sourcedetection.rst

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utilities/utils.rst


Development
===========
.. toctree::
   :maxdepth: 2
   :caption: Developers

   development.md
