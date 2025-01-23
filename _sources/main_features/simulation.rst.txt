karabo.simulation
=================

Overview
------------
This module contains the functionality required for an interferometer simulation. This includes defining a sky model and selecting a telescope. 

Classes
-------

.. autoclass:: karabo.simulation.sky_model.SkyModel
   :members:
   :special-members: __init__
   :exclude-members: __update_sky_model, __convert_ra_dec_to_cartesian

.. autoclass:: karabo.simulation.beam.BeamPattern
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta

.. autoclass:: karabo.simulation.observation.Observation
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta

.. autoclass:: karabo.simulation.observation.ObservationLong
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta

.. autoclass:: karabo.simulation.telescope.Telescope
   :members:
   :special-members: __init__
   :exclude-members: __update_sky_model, __convert_ra_dec_to_cartesian

.. autoclass:: karabo.simulation.interferometer.InterferometerSimulation
   :members:
   :special-members: __init__
   :exclude-members: __strfdelta, __run_simulation_oskar, __run_simulation_long, __get_OSKAR_settings_tree
