import copy
from abc import ABC
from datetime import datetime, timedelta
from itertools import cycle
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from karabo.error import KaraboError
from karabo.util._types import IntFloat, OskarSettingsTreeType


class ObservationAbstract(ABC):
    """Abstract base class for observations

    Holds all important information about an observation.
    """

    def __init__(
        self,
        *,
        mode: str = "Tracking",
        start_frequency_hz: IntFloat = 0,
        start_date_and_time: Union[datetime, str],
        length: timedelta = timedelta(hours=4),
        number_of_channels: int = 1,
        frequency_increment_hz: IntFloat = 0,
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
    ) -> None:
        """

        Args:
            start_date_and_time (Union[datetime, str]): Start time UTC and date for
                the observation. Strings are converted to datetime objects
                using `datetime.fromisoformat`.
            mode (str, optional): TODO. Defaults to "Tracking".
            start_frequency_hz (IntFloat, optional): The frequency at the start of the
                first channel in Hz.Defaults to 0.
            length (timedelta, optional): Length of observation.
                Defaults to timedelta(hours=4).
            number_of_channels (int, optional): Number of channels / bands to use.
                Defaults to 1.
            frequency_increment_hz (IntFloat, optional): Frequency increment between
                successive channels in Hz. Defaults to 0.
            phase_centre_ra_deg (IntFloat, optional): Right Ascension of
                the observation pointing (phase centre) in degrees.
                Defaults to 0.
            phase_centre_dec_deg (IntFloat, optional): Declination of the observation
                pointing (phase centre) in degrees.
                Defaults to 0.
            number_of_time_steps (int, optional): Number of time steps in the output
                data during the observation length. This corresponds to the number of
                correlator dumps for interferometer simulations, and the number of beam
                pattern snapshots for beam pattern simulations.
                Defaults to 1.
        """

        self.start_frequency_hz = start_frequency_hz

        if isinstance(start_date_and_time, str):
            self.start_date_and_time = datetime.fromisoformat(start_date_and_time)
        else:
            self.start_date_and_time = start_date_and_time

        self.length = length
        self.mode = mode

        # optional
        self.number_of_channels = number_of_channels
        self.frequency_increment_hz = frequency_increment_hz
        self.phase_centre_ra_deg = phase_centre_ra_deg
        self.phase_centre_dec_deg = phase_centre_dec_deg
        self.number_of_time_steps = number_of_time_steps

    def set_length_of_observation(
        self,
        hours: IntFloat,
        minutes: IntFloat,
        seconds: IntFloat,
        milliseconds: IntFloat,
    ) -> None:
        """
        Set a new length for the observation.
        Overriding the observation length set in the constructor.

        Args:
            hours: hours
            minutes: minutes
            seconds: seconds
            milliseconds: milliseconds
        """
        self.length = timedelta(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )

    def get_OSKAR_settings_tree(self) -> OskarSettingsTreeType:
        """
        Get the settings of this observation as an oskar setting tree.
        This function returns a python dictionary formatted
        according to the OSKAR documentation.
        `<https://fdulwich.github.io/oskarpy-doc/settings_tree.html>`.

        Returns:
            Dictionary containing the full configuration in the
            OSKAR Settings Tree format.
        """
        settings = {
            "observation": {
                "start_frequency_hz": str(self.start_frequency_hz),
                "mode": self.mode,
                # remove last three digits from milliseconds
                "start_time_utc": self.start_date_and_time.strftime(
                    "%d-%m-%Y %H:%M:%S.%f"
                )[:-3],
                "length": self.__strfdelta(self.length),
                "num_channels": str(self.number_of_channels),
                "frequency_inc_hz": str(self.frequency_increment_hz),
                "phase_centre_ra_deg": str(self.phase_centre_ra_deg),
                "phase_centre_dec_deg": str(self.phase_centre_dec_deg),
                "num_time_steps": str(self.number_of_time_steps),
            },
        }
        return settings

    @staticmethod
    def create_observations_oskar_from_lists(
        settings_tree: OskarSettingsTreeType,
        central_frequencies_hz: Union[IntFloat, List[IntFloat]],
        channel_bandwidths_hz: Union[IntFloat, List[IntFloat]],
        n_channels: Union[int, List[int]],
    ) -> List[OskarSettingsTreeType]:
        """
        Create observations for OSKAR settings from input lists.
        If there is a mix of different lengths of lists or single values,
        the missing information is repeated to match the longest list.

        Args:
        settings_tree : OskarSettingsTreeType
            The OSKAR settings tree, with 'observation' key among others.
        central_frequencies_hz : int or list of int
            List of central frequencies in MHz for each observation.
        channel_bandwidths_hz : int or list of int
            List of channel bandwidths in MHz for each observation.
        n_channels : int or list of int
            List of numbers of channels for each observation.

        Returns:
        list of dict: List of OSKAR observations,
        each as a dictionary with 'observation' key among others.

        Raises:
        ValueError: If the input lists are not of the same length.

        Note:
        The 'observation' key in each dictionary in the returned list has a value
        which is itself a dictionary, with keys 'start_frequency_hz',
        'num_channels', and 'frequency_inc_hz'.
        """
        # If the input is int, convert it into list
        if not isinstance(central_frequencies_hz, List):
            central_frequencies_hz = [central_frequencies_hz]
        if not isinstance(channel_bandwidths_hz, List):
            channel_bandwidths_hz = [channel_bandwidths_hz]
        if not isinstance(n_channels, List):
            n_channels = [n_channels]

        # Get max list length
        max_list_length = max(
            len(central_frequencies_hz), len(channel_bandwidths_hz), len(n_channels)
        )

        # Initialize cycle iterators
        cycle_cf = cycle(central_frequencies_hz)
        cycle_cb = cycle(channel_bandwidths_hz)
        cycle_nc = cycle(n_channels)

        # Extend the lists to match max_list_length by cycling over their elements
        while len(central_frequencies_hz) < max_list_length:
            central_frequencies_hz.append(next(cycle_cf))

        while len(channel_bandwidths_hz) < max_list_length:
            channel_bandwidths_hz.append(next(cycle_cb))

        while len(n_channels) < max_list_length:
            n_channels.append(next(cycle_nc))

        observations = []
        for cf, cb, nc in zip(
            central_frequencies_hz, channel_bandwidths_hz, n_channels
        ):
            obs = copy.deepcopy(settings_tree)
            obs["observation"]["start_frequency_hz"] = str(cf)
            obs["observation"]["num_channels"] = str(nc)
            obs["observation"]["frequency_inc_hz"] = str(cb)
            observations.append(obs)
        assert len(observations) == max_list_length

        return observations

    def __strfdelta(
        self,
        tdelta: timedelta,
    ) -> str:
        hours = tdelta.seconds // 3600 + tdelta.days * 24
        rm = tdelta.seconds % 3600
        minutes = rm // 60
        seconds = rm % 60
        milliseconds = tdelta.microseconds // 1000
        return "{}:{}:{}:{}".format(hours, minutes, seconds, milliseconds)

    def get_phase_centre(self) -> List[float]:
        return [self.phase_centre_ra_deg, self.phase_centre_dec_deg]

    def compute_hour_angles_of_observation(self) -> NDArray[np.float_]:
        """
        Given a total observation length and an integration time interval,
        determine the corresponding hour angles of observation.
        This utility function is used during simulations using the RASCIL backend.
        Approach based on https://gitlab.com/ska-sdp-china/rascil/-/blob/9002d853b64465238177b37e941c7445fed50d35/examples/performance/mid_write_ms.py#L32-40 # noqa: E501
        """
        total_observation_length = self.length
        integration_time = timedelta(
            seconds=self.length.total_seconds() / self.number_of_time_steps
        )

        if self.number_of_time_steps == 1:
            # If both times are the same, we create one observation
            # at hour angle = 0 that lasts integration_time seconds
            hour_angles = np.array([0])
        else:
            hour_angles = np.arange(
                int(-0.5 * total_observation_length.total_seconds()),
                int(0.5 * total_observation_length.total_seconds()),
                int(integration_time.total_seconds()),
            ) * (2 * np.pi / timedelta(days=1).total_seconds())

        return hour_angles


class Observation(ObservationAbstract):
    ...


class ObservationLong(ObservationAbstract):
    """
    This class allows the use of several observations on different
    days over a certain period of time within one day.
    If only ONE observation is desired, even if it takes a little longer,
    this is already possible using `Observation`.
    This class extends `Observation` so its parameters (except `length`)
    are not discussed here.
    `length` is little different, which describes the duration of ONE observation,
    whose maximum duration for `ObservationLong` is 24h.

    Args:
        number_of_days: Number of successive days to observe
    """

    def __init__(
        self,
        *,
        mode: str = "Tracking",
        start_frequency_hz: IntFloat = 0,
        start_date_and_time: Union[datetime, str],
        length: timedelta = timedelta(hours=4),
        number_of_channels: int = 1,
        frequency_increment_hz: IntFloat = 0,
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
        number_of_days: int = 2,
    ) -> None:
        self.enable_check = False
        super().__init__(
            mode=mode,
            start_frequency_hz=start_frequency_hz,
            start_date_and_time=start_date_and_time,
            length=length,
            number_of_channels=number_of_channels,
            frequency_increment_hz=frequency_increment_hz,
            phase_centre_ra_deg=phase_centre_ra_deg,
            phase_centre_dec_deg=phase_centre_dec_deg,
            number_of_time_steps=number_of_time_steps,
        )
        self.number_of_days: int = number_of_days
        self.__check_attrs()

    def __check_attrs(self) -> None:
        if not isinstance(self.number_of_days, int):
            raise KaraboError(
                "`number_of_days` must be of type int but "
                + f"is of type {type(self.number_of_days)}!"
            )
        if self.number_of_days <= 1:
            raise KaraboError(
                f"`number_of_days` must be >=2 but is {self.number_of_days}!"
            )
        if self.length > timedelta(hours=12):
            raise KaraboError(f"`length` should be max 12 hours but is {self.length}!")


class ObservationParallelized(ObservationAbstract):
    """
    This class allows the use of several observations on different
    days over a certain period of time within one day.
    If only ONE observation is desired, even if it takes a little longer,
    this is already possible using `Observation`.
    This class extends `Observation` so its parameters (except `length`)
    are not discussed here.
    `length` is little different, which describes the duration of ONE observation,
    whose maximum duration for `ObservationLong` is 24h.

    Args:
        number_of_days: Number of successive days to observe
    """

    def __init__(
        self,
        *,
        mode: str = "Tracking",
        center_frequencies_hz: Union[IntFloat, List[IntFloat]] = 100e6,
        start_date_and_time: Union[datetime, str],
        length: timedelta = timedelta(hours=4),
        n_channels: Union[int, List[int]] = [0, 1, 2, 3, 4, 5],
        channel_bandwidths_hz: Union[IntFloat, List[IntFloat]] = [1],
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
    ) -> None:
        self.enable_check = False
        super().__init__(
            mode=mode,
            start_frequency_hz=100e6,
            start_date_and_time=start_date_and_time,
            length=length,
            number_of_channels=1,
            frequency_increment_hz=0,
            phase_centre_ra_deg=phase_centre_ra_deg,
            phase_centre_dec_deg=phase_centre_dec_deg,
            number_of_time_steps=number_of_time_steps,
        )
        self.center_frequencies_hz = center_frequencies_hz
        self.n_channels = n_channels
        self.channel_bandwidths_hz = channel_bandwidths_hz
