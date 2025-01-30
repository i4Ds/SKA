class EastNorthCoordinate:
    """
    Class defining an east-north-(up) coordinate.

    Args:
        x: east coordinate in meters
        y: north coordinate in meters
        z: up coordinate in meters
        x_error: east coordinate error in meters
        y_error: north coordinate error in meters
        z_error: up coordinate error in meters

    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0,
        x_error: float = 0,
        y_error: float = 0,
        z_error: float = 0,
    ):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.x_error: float = x_error
        self.y_error: float = y_error
        self.z_error: float = z_error
