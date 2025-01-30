from typing import Tuple, TypeVar, cast

import numpy as np
import xarray as xr
from astropy.constants import c
from numpy.typing import NDArray

from karabo.util._types import IntFloat

_T = TypeVar("_T", NDArray[np.float_], xr.DataArray, IntFloat)


def convert_z_to_frequency(z: _T) -> _T:
    """Turn given redshift into corresponding frequency (Hz) for 21cm emission.

    Args:
        z: Redshift values to be converted into frequencies.

    Returns:
        Frequencies corresponding to input redshifts.
    """

    return cast(_T, c.value / (0.21 * (1 + z)))


def convert_frequency_to_z(freq: _T) -> _T:
    """Turn given frequency (Hz) into corresponding redshift for 21cm emission.

    Args:
        freq: Frequency values to be converted into redshifts.

    Returns:
        Redshifts corresponding to input frequencies.
    """

    return cast(_T, (c.value / (0.21 * freq)) - 1)


def freq_channels(
    z_obs: _T,
    channel_num: int = 10,
    equally_spaced_freq: bool = True,
) -> Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], np.float_]:
    """
    Calculates the frequency channels from the redshifts.

    Args:
        z_obs: Observed redshifts from the HI sources.
        channel_num: Number of channels.
        equally_spaced_freq: If True (default), create channels
            equally spaced in frequency.
            If False, create channels equally spaced in redshift.

    Returns:
        - Redshift channels array,
        - frequency channels array (in Hz),
        - array of bin widths of frequency channel (in Hz), for convenience, \
        and middle frequency (in Hz)
    """
    z_start = np.min(z_obs)
    z_end = np.max(z_obs)

    freq_endpoints = convert_z_to_frequency(np.array([z_start, z_end]))

    freq_start, freq_end = cast(Tuple[np.float_, np.float_], freq_endpoints)

    freq_mid = freq_start + (freq_end - freq_start) / 2

    if equally_spaced_freq is True:
        freq_channels_array = np.linspace(
            freq_start,
            freq_end,
            channel_num + 1,
        )

        redshift_channels_array = convert_frequency_to_z(freq_channels_array)
    else:
        redshift_channels_array = np.linspace(
            np.amin(z_obs),
            np.amax(z_obs),
            channel_num + 1,
        )

        freq_channels_array = convert_z_to_frequency(redshift_channels_array)

    freq_bins = np.abs(np.diff(freq_channels_array))

    print("The frequency channel starts at:", freq_start, "Hz")
    print("The bin sizes of the freq channel are:", freq_bins, "Hz")

    return redshift_channels_array, freq_channels_array, freq_bins, freq_mid
