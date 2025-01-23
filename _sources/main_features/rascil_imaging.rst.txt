karabo.imaging.imager_rascil
============================

Overview
------------
This package summerizes tools and functions to be used with the imager
from the RASCIL backend. This backend allows both calculating a dirty
images and a cleaned image, respoectivley.


Classes
-------

.. autoclass:: karabo.imaging.imager_rascil.RascilDirtyImagerConfig
   :members:
   :special-members: __init__
   :exclude-members:


.. autoclass:: karabo.imaging.imager_rascil.RascilDirtyImager
   :members:
   :special-members: __init__
   :exclude-members: _update_header_after_resize


.. autoclass:: karabo.imaging.imager_rascil.RascilImageCleanerConfig
   :members:
   :special-members: __init__
   :exclude-members: _update_header_after_resize


.. autoclass:: karabo.imaging.imager_rascil.RascilImageCleaner
   :members:
   :special-members: __init__
   :exclude-members: _compute
