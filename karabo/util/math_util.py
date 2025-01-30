"""
A collection of mathematical functions, mainly used for creating virtual skies.
"""
import math
from math import ceil, cos, floor, pi, sin, sqrt
from typing import List, Literal, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

from karabo.util._types import (
    FloatLike,
    NPFloatInpBroadType,
    NPFloatLike,
    NPFloatOutBroadType,
)


def poisson_disc_samples(
    width: FloatLike,
    height: FloatLike,
    r: int,
    k: int = 5,
    ord: Union[None, float, Literal["fro", "nuc"]] = None,
) -> List[Tuple[float, float]]:
    """
    This function helps to generate a virtual sky. It does this by randomly
    sampling points using **Poisson-disc sampling**. This produces points
    that are tightly-packed, but no closer to each other than a specified
    minimum distance r. This is important when sampling on a sphere.
    There is more information on https://www.jasondavies.com/maps/random-points/

    Args:
        width (FloatLike): Width of the sampling area
        height (FloatLike): Height of the area
        r (int): Minimal distance to keep between points
        k (int, optional): How often the algorithm tries to fit a point.
            Higher value give a more dense pointcloud but the generation takes
            longer. Defaults to 5.
        ord (Union[None, float, Literal['fro', 'nuc]], optional): Which norm to use
            for calulating the distance. Options are
                - 'fro': Frobenius norm or
                - 'nuc' for infinity

    Note:
        Calculating the norm is based on numpynp.linalg.norm(). See there for more
        information about the norm.

    Returns:
        List[Tuple[float, float]]: A list of point coordinates
    """
    tau = 2 * pi
    cellsize = r / sqrt(2)
    random = np.random.rand
    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [(np.inf, np.inf)] * (grid_width * grid_height)

    def grid_coords(p: Tuple[float, float]) -> Tuple[int, int]:
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(
        p: Tuple[float, float],
        gx: int,
        gy: int,
    ) -> bool:
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g == (np.inf, np.inf):
                    continue
                if np.linalg.norm(np.array(p) - np.array(g), ord=ord) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    grid = [p for p in grid if p != (np.inf, np.inf)]
    return grid


def get_poisson_disk_sky(
    min_size: Tuple[FloatLike, FloatLike],
    max_size: Tuple[FloatLike, FloatLike],
    flux_min: FloatLike,
    flux_max: FloatLike,
    r: int = 10,
) -> NDArray[np.float_]:
    """
    Creates a virtual sky by creating random source with flux between
    flux_min and flux_max.

    Args:
        min_size (Tuple[FloatLike, FloatLike]): For sky min RA and Deg in degrees.
        max_size (Tuple[FloatLike, FloatLike]): Max. RA and Deg of sky.
        flux_min (FloatLike): The minimal flux of the sources. Although the unit here
            is arbitrary it is usuall Jy/s.
        flux_max (FloatLike): The maximal flux.
        r (int, optional): The minimal distance between points. Higher value gives
            sparser sky. Defaults to 10.

    Returns:
        NDArray[np.float_]: Array of sources and flux \
            [[x1, y1, flux], [x2, y2, flux] ...]
    """
    assert flux_max >= flux_min
    x = min_size[0]
    y = min_size[1]
    X = max_size[0]
    Y = max_size[1]
    width = cast(FloatLike, abs(X - x))
    height = cast(FloatLike, abs(Y - y))
    center_x = x + (X - x) * 0.5
    center_y = y + (Y - y) * 0.5
    samples = poisson_disc_samples(width, height, r)
    np_samples = np.array(samples)
    ra = np_samples[:, 0] - (width * 0.5)
    dec = np_samples[:, 1] - (height * 0.5)
    ra = ra + center_x
    dec = dec + center_y
    np_samples = np.vstack((ra, dec)).transpose()
    flux = np.random.random((len(samples), 1)) * (flux_max - flux_min) + flux_min
    sky_array = np.hstack((np_samples, flux))
    return sky_array


#
def long_lat_to_cartesian(lat: NPFloatLike, lon: NPFloatLike) -> NDArray[np.float_]:
    """
    Converts geodesic coordinates (latitude and longitude) into geocentric ones,
    also called cartesian coordinates.

    Args:
        lat (NPFloatLike): The latitude in degrees, west is negative.
        lon (NPFloatLike): The longitude to convert, south is negative.

    Returns:
        NDArray[np.float_]: An array with the geodesic coordinates [x, y, z]
    """
    lat_, lon_ = np.deg2rad(lat), np.deg2rad(lon)
    x = R * cos(lat_) * cos(lon_)
    y = R * cos(lat_) * sin(lon_)
    z = R * sin(lat_)
    out = cast(
        NDArray[np.float_], np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    )
    return out


#
#
# def cartesian_to_long_lat(cart: [float, float, float]):
#     lat = np.degrees(np.arcsin(cart[2]))
#     lon = np.degrees(np.arctan2(cart[1], cart[0]))
#     return np.array([lon, lat])


R = 6360000  # earth radius


def cartesian_to_ll(
    x: FloatLike,
    y: FloatLike,
    z: int = 0,
) -> Tuple[float, float]:
    """
    Converts cartesian coordinates (geocentric) of a point on Earth to its
    corresponding geodesic ones (latitude and longitude).

    Note:
        The converions does not take z into account. Instead, we set z to
        be the Earth radius.

    Args:
        x: cartesian x coordinate
        y: cartesian y coordinate
        z (Optional): cartesian z coordinate. Defaults to the Earth radius.

    Returns:
        Tuple[float, float]: Geodesic coordinates (latitude, longitude)
    """
    # does not use `z`
    r = math.sqrt(x**2 + y**2)
    long = 180 * math.atan2(y, x) / math.pi
    lat = 180 * math.acos(r / R) / math.pi
    return lat, long


def Gauss(
    x: NPFloatInpBroadType,
    x0: NPFloatInpBroadType,
    y0: NPFloatInpBroadType,
    a: NPFloatInpBroadType,
    sigma: NPFloatInpBroadType,
) -> NPFloatOutBroadType:
    """
    Calculates the value of the Gaussian distribution at a single point `x` or
    for an array of points.

    This function is used in `karabo.util.data_util.get_spectral_sky_data()` but it
    can be used elsewhere.

    Note:
        An argument can be a number or an array. If using arrays all shapes must match.

    Args:
        x (NPFloatInpBroadType): Where to calculate the value
        x0 (NPFloatInpBroadType): Center point of distribution
        y0 (NPFloatInpBroadType): Offset in y direction
        a (NPFloatInpBroadType): Amplitude (height of distribution function)
        sigma (NPFloatInpBroadType): standard deviation (width of distribution)

    Returns:
        NPFloatOutBroadType: A value or an array having the values of the \
            Gaussian funtion.
    """
    gauss = y0 + a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    return cast(NPFloatOutBroadType, gauss)


def Voigt(
    x: NPFloatInpBroadType,
    x0: NPFloatInpBroadType,
    y0: NPFloatInpBroadType,
    a: NPFloatInpBroadType,
    sigma: NPFloatInpBroadType,
    gamma: NPFloatInpBroadType,
) -> NPFloatOutBroadType:
    """
    Calculates the value of the Voigt profile at a single point `x` or
    for an array of points.

    This function is used in `karabo.util.data_util.get_spectral_sky_data()` but it
    can be used elsewhere.

    Note:
        An argument can be a number or an array. If using arrays all shapes must match.

    Args:
        x (NPFloatInpBroadType): Where to calculate the value
        x0 (NPFloatInpBroadType): Center point of distribution
        y0 (NPFloatInpBroadType): Offset in y direction
        a (NPFloatInpBroadType): Amplitude (height of distribution function)
        sigma (NPFloatInpBroadType): standard deviation (width of distribution)
        gamma (NPFloatInpBroadType): FWHM of Lorentz component

    Returns:
        NPFloatOutBroadType: A value or an array having the values of the \
            Gaussian funtion.
    """
    # sigma = alpha / np.sqrt(2 * np.log(2))
    voigt = y0 + a * np.real(
        wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))
    ) / sigma / np.sqrt(2 * np.pi)
    return cast(NPFloatOutBroadType, voigt)
