from __future__ import annotations

import copy
import enum
import math
import re
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from warnings import warn

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar
import pandas as pd
import xarray as xr
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.io.fits import ColDefs
from astropy.io.fits.fitsrec import FITS_rec
from astropy.table import Table
from astropy.units.core import PrefixUnit, Unit, UnitBase
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from numpy.typing import NDArray
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from typing_extensions import assert_never
from xarray.core.coordinates import DataArrayCoordinates

from karabo.data.external_data import (
    GLEAMSurveyDownloadObject,
    HISourcesSmallCatalogDownloadObject,
    MALSSurveyV3DownloadObject,
    MIGHTEESurveyDownloadObject,
)
from karabo.error import KaraboSkyModelError
from karabo.simulation.line_emission_helpers import (
    convert_frequency_to_z,
    convert_z_to_frequency,
)
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import (
    FilePathType,
    IntFloat,
    IntFloatList,
    NPFloatInpBroadType,
    NPFloatLike,
    PrecisionType,
)
from karabo.util.hdf5_util import convert_healpix_2_radec, get_healpix_image
from karabo.util.math_util import get_poisson_disk_sky
from karabo.util.plotting_util import get_slices
from karabo.warning import _DEV_ERROR_MSG, KaraboWarning

StokesType = Literal["Stokes I", "Stokes Q", "Stokes U", "Stokes V"]
SkySourcesColName = Literal[  # preserve python-var-name compatibility
    "ra",
    "dec",
    "stokes_i",
    "stokes_q",
    "stokes_u",
    "stokes_v",
    "ref_freq",
    "spectral_index",
    "rm",
    "major",
    "minor",
    "pa",
    "true_redshift",
    "observed_redshift",
    "id",
]

_NPSkyType = Union[NDArray[np.float_], NDArray[np.object_]]
_SkySourcesType = Union[_NPSkyType, xr.DataArray]
_SourceIdType = Union[
    List[str],
    List[int],
    List[float],
    NDArray[np.object_],
    NDArray[np.int_],
    NDArray[np.float_],
    DataArrayCoordinates[xr.DataArray],
]
_SkyPrefixMappingValueType = Union[str, List[str]]
_ChunksType = Union[
    int,
    Literal["auto"],
    Tuple[int, ...],
    Mapping[Hashable, int],
]
_TSkyModel = TypeVar("_TSkyModel", bound="SkyModel")


class Polarisation(enum.Enum):
    STOKES_I = (0,)
    STOKES_Q = (1,)
    STOKES_U = (2,)
    STOKES_V = 3


@dataclass
class SkyPrefixMapping:
    """Defines the relation between col-names of a .fits file and `SkyModel.sources`.

    Field-names of sources-data must be according to `SkySourcesColName`.
    Each `_SkyPrefixMappingValueType` can be a set of col-names, a single col-name.
    Just leave the values not provided in the .fits file None.
    """

    # field-names must match `SkySourcesColName`
    ra: _SkyPrefixMappingValueType
    dec: _SkyPrefixMappingValueType
    stokes_i: _SkyPrefixMappingValueType
    stokes_q: Optional[_SkyPrefixMappingValueType] = None
    stokes_u: Optional[_SkyPrefixMappingValueType] = None
    stokes_v: Optional[_SkyPrefixMappingValueType] = None
    ref_freq: Optional[str] = None  # providing multiple freq-col-names is not supported
    spectral_index: Optional[_SkyPrefixMappingValueType] = None
    rm: Optional[_SkyPrefixMappingValueType] = None
    major: Optional[_SkyPrefixMappingValueType] = None
    minor: Optional[_SkyPrefixMappingValueType] = None
    pa: Optional[_SkyPrefixMappingValueType] = None
    true_redshift: Optional[_SkyPrefixMappingValueType] = None
    observed_redshift: Optional[_SkyPrefixMappingValueType] = None
    id: Optional[_SkyPrefixMappingValueType] = None


@dataclass
class SkySourcesUnits:
    """Represents the units of `SkyModel.sources`.

    This class is useful for unit-conversion from different Prefixes
    https://docs.astropy.org/en/stable/units/standard_units.html
    and different units which can be converted to another (like deg to arcmin).

    `UnitBase` covers
    - `Unit`: e.g. u.Jy, u.Hz
    - `CompositeUnit`: e.g. u.Jy/u.beam, u.rad/u.m**2
    - `PrefixUnit`: e.g. u.MHz, u.GHz

    Just assign another unit to the constructor in case the default doesn't fit
    for a specific field. It's useful if e.g. `stokes_i` is u.Jy/u.beam in the
    .fits file instead.
    """

    # field-names of unit-values must match `SkySourcesColName`
    ra: UnitBase = u.deg  # `Unit`
    dec: UnitBase = u.deg  # `Unit`
    stokes_i: UnitBase = u.Jy  # `Unit`
    stokes_q: UnitBase = u.Jy  # `Unit`
    stokes_u: UnitBase = u.Jy  # `Unit`
    stokes_v: UnitBase = u.Jy  # `Unit`
    ref_freq: UnitBase = u.Hz  # `Unit`
    rm: UnitBase = u.rad / u.m**2  # `CompositeUnit`
    major: UnitBase = u.arcsec  # `Unit`
    minor: UnitBase = u.arcsec  # `Unit`
    pa: UnitBase = u.deg  # `Unit`

    @classmethod
    def format_sky_prefix_freq_mapping(
        cls,
        cols: ColDefs,
        prefix_mapping: SkyPrefixMapping,
        encoded_freq: Optional[UnitBase],
    ) -> Tuple[SkyPrefixMapping, int, Dict[str, float]]:
        """Formats `SkyPrefixMapping` fields from str to list[str] if formattable.

        The encoded frequencies and col-names of the formattable field-values get
        extracted from `cols`, converted into Hz. The extracted col-names and
        frequencies are then available in a col-name -> freq mapping.
        In addition, `prefix_mapping` gets enhanced with a list of according col-names
        of the .fits file.
        This function doesn't do any formatting for each field, if `encoded_freq`
        is None or is not positional formattable.

        Args:
            cols: Columns of .fits file
            prefix_mapping: Should contain formattable field-values.
            encoded_freq: astropy.unit frequency encoded (e.g. u.MHz)

        Raises:
            RuntimeError: In case the number of formatting columns of `cols`
                are not of equal for each frequency-channel.

        Returns:
            Formatted `prefix_mapping`, number of formatted col-names per frequency,
                and a mapping col-name -> frequency (Hz).
        """
        num_formattings = 0
        names_and_freqs: Dict[str, float] = dict()
        if encoded_freq is None:
            return prefix_mapping, num_formattings, names_and_freqs
        prefix_mapping_copied = deepcopy(prefix_mapping)  # to preserve mutable object
        for field in fields(prefix_mapping_copied):
            field_value = getattr(prefix_mapping_copied, field.name)
            if isinstance(field_value, str) and cls.is_pos_formattable(field_value):
                col_names_and_freqs = cls.extract_names_and_freqs(
                    string=field_value, cols=cols, unit=encoded_freq
                )
                if num_formattings == 0:
                    num_formattings = len(col_names_and_freqs)
                else:
                    if num_formattings != len(col_names_and_freqs):
                        raise RuntimeError(
                            "Number of formatted sky-prefixes don't match for each "
                            + f"frequency-channel! {num_formattings=} != "
                            + f"{len(col_names_and_freqs)=}"
                        )
                names_and_freqs.update(col_names_and_freqs)
                setattr(
                    prefix_mapping_copied,
                    field.name,
                    [field_name for field_name in col_names_and_freqs],
                )
        if encoded_freq is not None and num_formattings == 0:
            raise RuntimeError(
                "Providing `encoded_freq` with non-formattable column-names "
                + "is not allowed!"
            )
        return prefix_mapping_copied, num_formattings, names_and_freqs

    def get_unit_scales(
        self,
        cols: ColDefs,
        unit_mapping: Dict[str, UnitBase],
        prefix_mapping: SkyPrefixMapping,
    ) -> Dict[str, float]:
        """Converts all units specified in `prefix_mapping` into a factor, which can be
        used for multiplication of the according data-array, to convert the
        `SkyModel.sources` into standard-units defined in the fields of this dataclass.

        If a unit in the .fits file doesn't match the default units of this dataclass,
        just change it during instantiation to match the .fits file unit.

        Args:
            cols: Columns from `hdul[1].columns` .fits file.
            unit_mapping: `Mapping from col-unit (from .fits file) to `astropy.units`.
                Be aware to use the very-same unit. E.g. if it's prefixed in the .fits,
                also prefix in the `astropy.units` ("MHZ":u.MHz, NOT "MHZ":u.Hz).
            prefix_mapping: Mapping from `SkyModel.sources` to col-name(s).


        Returns:
            Mapping col-name -> scaling-factor to `SkyModel.sources` standard units.
        """
        used_units: List[str] = list()
        used_col_names: List[str] = list()
        cols_to_field_name_of_interest: Dict[str, str] = dict()
        for field in fields(prefix_mapping):
            if (
                col_names_of_interest := getattr(prefix_mapping, field.name)
            ) is not None:
                if isinstance(col_names_of_interest, list):
                    for col_name_of_interest in col_names_of_interest:
                        cols_to_field_name_of_interest[
                            col_name_of_interest
                        ] = field.name
                elif isinstance(col_names_of_interest, str):
                    cols_to_field_name_of_interest[col_names_of_interest] = field.name
                else:
                    # should never happen, but `getattr` has no type-hints
                    raise TypeError(
                        f"{type(col_names_of_interest)=} must be list[str] | str."
                    )
        scales: Dict[str, float] = dict()
        for col in cols:
            if (col_name := col.name) in cols_to_field_name_of_interest.keys():
                used_col_names.append(col_name)  # here because of cols without units
                if (unit := col.unit) is not None:
                    if unit not in used_units:
                        used_units.append(unit)
                    astropy_unit_of_fits_col = unit_mapping[unit]
                    # here, it assumes that unit-field-names of `SkySourcesUnits` and
                    # `SkyPrefixMapping` are the same
                    sky_sources_unit: UnitBase = getattr(
                        self, cols_to_field_name_of_interest[col_name]
                    )
                    scales[col_name] = astropy_unit_of_fits_col.to(sky_sources_unit)

        non_used_unit_keys = set(unit_mapping.keys()) - set(used_units)
        if len(non_used_unit_keys) > 0:
            # here, we double check all columns, not just the specified to be more
            # robust with exception-throwing
            not_found_unit_keys: List[str] = list()
            for non_used_unit_key in non_used_unit_keys:
                found = False
                for col in cols:
                    if col.unit == non_used_unit_key:
                        found = True
                        break
                if not found:
                    not_found_unit_keys.append(non_used_unit_key)
            if len(not_found_unit_keys) > 0:
                raise RuntimeError(
                    "The following unit-strings defined in `unit_mapping` are not "
                    + f"found in `cols`, which may be a typo: {not_found_unit_keys}"
                )
        non_used_col_names = set(cols_to_field_name_of_interest.keys()) - set(
            used_col_names
        )
        if len(non_used_col_names) > 0:
            raise RuntimeError(
                "The following col-names defined in `prefix_mapping` were not found "
                + f"in `cols`, which may be a typo: {non_used_col_names}"
            )
        return scales

    @classmethod
    def is_pos_formattable(cls, string: str) -> bool:
        """Checks if `string` is positional formattable.

        It is positional-formattable if `string` contains a '{0}'.

        Args:
            string: str to check.

        Returns:
            Result if it's formattable.
        """
        return "{0}" in string

    @classmethod
    def extract_names_and_freqs(
        cls,
        string: str,
        cols: ColDefs,
        unit: Union[Unit, PrefixUnit],
    ) -> Dict[str, float]:
        """Extracts all col-names and it's encoded frequency (in Hz).

        `string` has to be formattable, meaning it must contain a {0} for being
        the frequency placeholder. Otherwise it will fail.

        Important: This function doesn't consider the unit of `SkyModel.sources`
        "ref_freq" unit. It converts it to Hz regardless.

        Args:
            string: Column-name with {0} frequency-placeholder.
            cols: Columns of the .fits file to search through.
            unit: Frequency-unit encoded in col-names (e.g. u.MHz).

        Returns:
            Mapping col-name -> decoded-frequency in Hz.
        """
        if not cls.is_pos_formattable(string=string):
            raise ValueError(f"{string=}" + "must contain '{0}' to be formattable!")
        number_format = r"(\d+)"  # just supports int-numbers atm
        pattern = re.compile(f"^{string.format(number_format)}$")
        names_and_freqs: Dict[str, float] = dict()
        for col in cols:
            col_name = col.name
            if not isinstance(col_name, str):
                raise TypeError(
                    f"`col_name` should be of type `str`, but is {type(col_name)=}"
                )
            match = pattern.match(string=col_name)
            if match:
                extracted_freq = float(match.group(1))
                extracted_freq = float((extracted_freq * unit).to(u.Hz).value)  # type: ignore[attr-defined] # noqa: E501
                names_and_freqs[col_name] = extracted_freq
        return names_and_freqs

    @classmethod
    def get_pos_ids_to_ra_dec(
        cls,
        pos_ids: Union[Sequence[str], str],
    ) -> Dict[str, Tuple[float, float]]:
        """Converts position-id(s) from str to ra-dec in degrees.

        The supported format(s) of `pos_ids` is 'JHHMMSS(.s+)±DDMMSS(.s+)' (J2000).
        The format is not tested in it's entirety to not have to do a sanity-check
        for each pos-id.

        A valid example of `pos_ids` (here just str instead of an iterable) is
        `pos_ids`="J130508.50-285042.0", which results in RA≈196.285, DEC≈-28.845

        Args:
            pos_ids: Position-id(s)

        Returns:
            Dict from pos-id to ra-dec-tuple.
        """
        if isinstance(pos_ids, str):
            pos_ids = [pos_ids]

        def get_sign(pos_id: str) -> str:
            """Extracts the pos-id separator.

            Allowed separators are '+' and '-'.

            Args:
                pos_id: Position-id to extract sign-separator from.

            Returns:
                Sign separator.
            """
            if "+" in pos_id:
                sign = "+"
            elif "-" in pos_id:
                sign = "-"
            else:
                raise ValueError(
                    f"{pos_id} doesn't contain a valid sign-separator '+' or '-'"
                )
            return sign

        def create_angle_str(formatted_pos: str) -> str:
            """Creates `astropy.coordinates.Angle` compatible string.

            Assumes '+'|'-'|'J' at the start and consecutive 2x2x2(.n) `angle_values`.

            Args:
                formatted_pos: RA or DEC string from a single pos-id.

            Returns:
                Angle string.
            """
            if (first_char := formatted_pos[0]) == "+" or first_char == "-":
                angle_str = first_char
                angle_values = ("d", "m", "s")
            elif first_char == "J":
                angle_str = ""
                angle_values = ("h", "m", "s")
            else:
                raise ValueError(f"{formatted_pos=} must start with 'J', '+' or '-'!")
            if (
                len(formatted_pos.split(".")[0]) < 7
            ):  # min: prefix + 2x2x2 `angle_values`
                raise ValueError(
                    f"{formatted_pos=} doesn't follow the format "
                    + "(J|+|-)(DD|HH)MMSS(.s+)*"
                )
            unit_indices = [
                (i * 2 + 1, i * 2 + 3) if value != "s" else (i * 2 + 1, None)
                for i, value in enumerate(angle_values)
            ]
            for unit_char, unit_idxs in zip(angle_values, unit_indices):
                angle_str += f"{formatted_pos[unit_idxs[0]:unit_idxs[1]]}{unit_char}"
            return angle_str

        def convert_pos_id_to_ra_dec(pos_id: str) -> tuple[float, float]:
            """Converts a single pos-id to ra-dec tuple in degrees.

            Assumes that unit-matches for each unit are all consecutive.

            Args:
                pos_id: Positional-id to convert.

            Returns:
                RA-DEC tuple in deg.
            """
            sign = get_sign(pos_id=pos_id)
            ra_pos, dec_pos = pos_id.split(sign, maxsplit=1)
            dec_pos = f"{sign}{dec_pos}"  # keeps sign in dec-str
            ra_angle_str = create_angle_str(formatted_pos=ra_pos)
            dec_angle_str = create_angle_str(formatted_pos=dec_pos)
            ra = Angle(ra_angle_str).to(u.deg).value
            dec = Angle(dec_angle_str).to(u.deg).value
            return ra, dec

        ra_dec: Dict[str, Tuple[float, float]] = dict()
        for pos_id in pos_ids:
            ra_dec[pos_id] = convert_pos_id_to_ra_dec(
                pos_id=pos_id,
            )
        return ra_dec


XARRAY_DIM_0_DEFAULT, XARRAY_DIM_1_DEFAULT = cast(
    Tuple[str, str], xr.DataArray([[]]).dims
)


class SkyModel:
    """
    Class containing all information of the to be observed Sky.

    `SkyModel.sources` is a `xarray.DataArray`
    (https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html).
    `np.ndarray` are also supported as input type for `SkyModel.sources`,
    however, the values in `SkyModel.sources are converted to `xarray.DataArray`.

    `SkyModel.compute` method is used to load the data into memory as a numpy array.
    It should be called after all the filtering and other operations are completed
    and to avoid doing the same calculation multiple them when e.g. on a cluster.

    Args:
        sources: List of all point sources in the sky as `xarray.DataArray`.
            The source_ids reside in `SkyModel.source_ids` if provided
            through `xarray.sources.coords` with an arbitrary string key
            as index or `np.ndarray` as idx SOURCES_COLS.
            A single point source is described using the following col-order:

            - [0] right ascension (deg)
            - [1] declination (deg)
            - [2] stokes I Flux (Jy)
            - [3] stokes Q Flux (Jy): defaults to 0
            - [4] stokes U Flux (Jy): defaults to 0
            - [5] stokes V Flux (Jy): defaults to 0
            - [6] reference_frequency (Hz): defaults to 0
            - [7] spectral index (N/A): defaults to 0
            - [8] rotation measure (rad / m^2): defaults to 0
            - [9] major axis FWHM (arcsec): defaults to 0
            - [10] minor axis FWHM (arcsec): defaults to 0
            - [11] position angle (deg): defaults to 0
            - [12] true redshift: defaults to 0
            - [13] observed redshift: defaults to 0
            - [14] object-id: just for `np.ndarray` \
            it is removed in the `xr.DataArray`
            and exists then in `xr.DataArray.coords` as index.
        wcs: World Coordinate System (WCS) object representing the coordinate
            transformation between pixel coordinates and celestial coordinates
            (e.g., right ascension and declination).
        precision: The precision of numerical values used in the SkyModel.
            Has to be of type np.float_.
        h5_file_connection: An open connection to an HDF5 (h5) file
            that can be used to store or retrieve data related to the SkyModel.
    """

    SOURCES_COLS = 14
    _STOKES_IDX: Dict[StokesType, int] = {
        "Stokes I": 2,
        "Stokes Q": 3,
        "Stokes U": 4,
        "Stokes V": 5,
    }
    COL_IDX: Dict[SkySourcesColName, int] = {  # according to docstring
        "ra": 0,
        "dec": 1,
        "stokes_i": 2,
        "stokes_q": 3,
        "stokes_u": 4,
        "stokes_v": 5,
        "ref_freq": 6,
        "spectral_index": 7,
        "rm": 8,
        "major": 9,
        "minor": 10,
        "pa": 11,
        "true_redshift": 12,
        "observed_redshift": 13,
        "id": 14,
    }
    COL_NAME: Dict[int, SkySourcesColName] = dict(zip(COL_IDX.values(), COL_IDX.keys()))

    def __init__(
        self,
        sources: Optional[_SkySourcesType] = None,
        wcs: Optional[WCS] = None,
        precision: Type[np.float_] = np.float64,
        h5_file_connection: Optional[h5py.File] = None,
    ) -> None:
        """
        Initialize a SkyModel object.

        Args:
            sources : {xarray.DataArray, np.ndarray}, optional
                List of all point sources in the sky.
                It can be provided as an `xarray.DataArray` or `np.ndarray`.
                If provided as an `np.ndarray`, the values are converted to
                `xarray.DataArray`.
            wcs : WCS, optional
                World Coordinate System (WCS) object representing the coordinate
                transformation between pixel coordinates and celestial coordinates.
            precision : np.dtype, optional
                The precision of numerical values used in the SkyModel.
                It should be a NumPy data type (e.g., np.float64).
            h5_file_connection : h5py.File, optional
                An open connection to an HDF5 (h5) file
                that can be used to store or retrieve data related to the SkyModel.
        """
        self.__sources_dim_sources = XARRAY_DIM_0_DEFAULT
        self.__sources_dim_data = XARRAY_DIM_1_DEFAULT
        self._sources: Optional[xr.DataArray] = None
        self.precision = precision
        self.wcs = wcs
        self.sources = sources  # type: ignore [assignment]
        self.h5_file_connection = h5_file_connection

    def __get_empty_sources(self, n_sources: int) -> xr.DataArray:
        empty_sources = np.hstack((np.zeros((n_sources, SkyModel.SOURCES_COLS)),))
        return xr.DataArray(
            empty_sources, dims=[self._sources_dim_sources, self._sources_dim_data]
        )

    def __set_sky_xarr_dims(self, sources: _SkySourcesType) -> None:
        if isinstance(sources, np.ndarray):
            pass  # nothing to do here
        elif isinstance(sources, xr.DataArray):  # checks xarray dims through setter
            self._sources_dim_sources, self._sources_dim_data = cast(
                Tuple[str, str], sources.dims
            )
        else:
            assert_never(f"{type(sources)} is not a valid `SkySourcesType`.")

    def close(self) -> None:
        """
        Closes the connection to the HDF5 file if it is open and
        sets the `h5_file_connection` attribute to `None`.
        """
        if self.h5_file_connection:
            self.h5_file_connection.close()
            self.h5_file_connection = None

    @staticmethod
    def copy_sky(sky: _TSkyModel) -> _TSkyModel:
        if sky.h5_file_connection is not None:
            h5_connection = sky.h5_file_connection
            sky.h5_file_connection = None
        else:
            h5_connection = None

        copied_sky = copy.deepcopy(sky)
        if h5_connection is not None:
            copied_sky.h5_file_connection = h5_connection

        return copied_sky

    def compute(self) -> None:
        """
        Loads the lazy data into a numpy array which is then
        wrapped in a `xarray.DataArray`. It performs the computation
        necessary to obtain the actual data and stores it in the `_sources` attribute.
        After the computation is complete,
        it calls the `close` method to close the connection to the HDF5 file.

        Returns:
            None
        """
        if self.sources is not None:
            computed_sources = self.sources.compute()
        self.sources = None
        self.sources = computed_sources
        self.close()

    def _check_sources(self, sources: _SkySourcesType) -> None:
        self.__set_sky_xarr_dims(sources=sources)
        if isinstance(sources, np.ndarray):
            pass
        elif isinstance(sources, xr.DataArray):
            if self.sources is None:
                return None
            if self.sources.indexes.keys() != sources.indexes.keys():
                sky_keys = list(self.sources.indexes.keys())
                provided_keys = list(sources.indexes.keys())
                raise KaraboSkyModelError(
                    f"Provided index name {provided_keys} (object id's) of `sources` "
                    + f"don't match already existing index name {sky_keys}."
                )
        else:
            assert_never(f"{type(sources)} is not a valid `SkySourcesType`.")
        return None

    def to_sky_xarray(self, sources: _SkySourcesType) -> xr.DataArray:
        """Converts a `np.ndarray` or `xr.DataArray` to `SkyModel.sources`
            compatible `xr.DataArray`.

        Args:
            sources: Array to convert. Col-order see `SkyModel` description.

        Returns:
            `SkyModel.sources` compatible `xr.DataArray`.
        """
        if len(sources.shape) != 2:
            raise KaraboSkyModelError(
                "`sources` must be 2-dimensional but "
                + f"is {len(sources.shape)}-dimensional."
            )
        if sources.shape[1] < 3:
            raise KaraboSkyModelError(
                "`sources` requires min 3 cols: `right_ascension`, "
                + "`declination` and `stokes I flux`."
            )
        if isinstance(sources, xr.DataArray):
            if sources.shape[1] < SkyModel.SOURCES_COLS:
                fill = self.__get_empty_sources(n_sources=sources.shape[0])
                if len(sources.coords) > 0:
                    fill.coords[self._sources_dim_sources] = sources.coords[
                        self._sources_dim_sources
                    ]
                missing_cols = SkyModel.SOURCES_COLS - sources.shape[1]
                fill[:, :-missing_cols] = sources
                da = fill
            else:
                da = sources
        elif isinstance(sources, np.ndarray):
            # For numpy ndarray, we delete the ID column of the sources
            if sources.shape[1] in (
                1 + SkyModel.SOURCES_COLS,
                13,
            ):  # sources have IDs. 13 is for backwards compatibility
                index_of_ids_column = sources.shape[1] - 1
                source_ids = sources[:, index_of_ids_column]
                sources = np.delete(sources, np.s_[index_of_ids_column], axis=1)  # type: ignore [assignment] # noqa: E501
                sources = sources.astype(self.precision)
                da = xr.DataArray(
                    sources,
                    dims=[self._sources_dim_sources, self._sources_dim_data],
                    coords={self._sources_dim_sources: source_ids},
                )
            elif sources.shape[1] == SkyModel.SOURCES_COLS:
                da = xr.DataArray(
                    sources,
                    dims=[self._sources_dim_sources, self._sources_dim_data],
                )
            else:
                fill = self.__get_empty_sources(n_sources=sources.shape[0])
                missing_cols = SkyModel.SOURCES_COLS - sources.shape[1]
                fill[:, :-missing_cols] = sources
                da = fill
        else:
            assert_never(f"{type(sources)} is not a valid `SkySourcesType`.")

        return da

    def add_point_sources(self, sources: _SkySourcesType) -> None:
        """Add new point sources to the sky model.

        Args:
            sources: `np.ndarray` with shape (number of sources, 1 + SOURCES_COLS),
                where you can place the "source_id" at index SOURCES_COLS.
                OR an `xarray.DataArray` with shape (number of sources, SOURCES_COLS),
                where you can place the "source_id" at `xarray.DataArray.coord`
                or use `SkyModel.source_ids` later.

                The column indices correspond to:

                - [0] right ascension (deg)
                - [1] declination (deg)
                - [2] stokes I Flux (Jy)
                - [3] stokes Q Flux (Jy): defaults to 0
                - [4] stokes U Flux (Jy): defaults to 0
                - [5] stokes V Flux (Jy): defaults to 0
                - [6] reference_frequency (Hz): defaults to 0
                - [7] spectral index (N/A): defaults to 0
                - [8] rotation measure (rad / m^2): defaults to 0
                - [9] major axis FWHM (arcsec): defaults to 0
                - [10] minor axis FWHM (arcsec): defaults to 0
                - [11] position angle (deg): defaults to 0
                - [12] true redshift: defaults to 0
                - [13] observed redshift: defaults to 0
                - [14] source id (object): is in `SkyModel.source_ids` if provided

        """
        if (len(sources.shape) == 2) and (sources.shape[0] == 0):
            warn(
                """There are no sources in the received sources array.
            Will not modify the current SkyModel instance."""
            )
            return
        try:
            sds, sdd = self._sources_dim_sources, self._sources_dim_data
            self._check_sources(sources=sources)
            sky_sources = self.to_sky_xarray(sources=sources)
            if sky_sources.shape[1] > SkyModel.SOURCES_COLS:
                sky_sources[:, : SkyModel.SOURCES_COLS] = sky_sources[
                    :, : SkyModel.SOURCES_COLS
                ].astype(self.precision)
            if self.sources is not None:
                self.sources = xr.concat(
                    (self.sources, sky_sources), dim=self._sources_dim_sources
                )
            else:
                self._sources = sky_sources
        except BaseException:  # rollback of dim-names if sth goes wrong
            self._sources_dim_sources, self._sources_dim_data = sds, sdd
            raise

    def write_to_file(self, path: str) -> None:
        self.save_sky_model_as_csv(path)

    @classmethod
    def read_from_file(cls: Type[_TSkyModel], path: str) -> _TSkyModel:
        """
        Read a CSV file in to create a SkyModel.
        The CSV should have the following columns

        - right ascension (deg)
        - declination (deg)
        - stokes I Flux (Jy)
        - stokes Q Flux (Jy): if no information available, set to 0
        - stokes U Flux (Jy): if no information available, set to 0
        - stokes V Flux (Jy): if no information available, set to 0
        - reference_frequency (Hz): if no information available, set to 0
        - spectral index (N/A): if no information available, set to 0
        - rotation measure (rad / m^2): if no information available, set to 0
        - major axis FWHM (arcsec): if no information available, set to 0
        - minor axis FWHM (arcsec): if no information available, set to 0
        - position angle (deg): if no information available, set to 0
        - true redshift: defaults to 0
        - observed redshift: defaults to 0
        - source id (object): is in `SkyModel.source_ids` if provided

        Args:
            path: file to read in
        Returns:
            SkyModel
        """
        dataframe = pd.read_csv(path).to_numpy()

        if dataframe.shape[1] < 3:
            raise KaraboSkyModelError(
                "CSV does not have the necessary 3 basic columns (RA, DEC and "
                f"STOKES I), but only {dataframe.shape[1]} columns."
            )

        if dataframe.shape[1] > cls.SOURCES_COLS:
            print(
                f"CSV has {dataframe.shape[1] - cls.SOURCES_COLS + 1} "
                "rows too many. The extra rows will be cut off."
            )

        return cls(dataframe)

    def to_np_array(self, with_obj_ids: bool = False) -> _NPSkyType:
        """
        Gets the sources as np.ndarray

        Args:
            with_obj_ids: Option whether object ids should be included or not

        Returns:
            the sources of the SkyModel as np.ndarray
        """
        if self.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `to_np_array`."
            )
        if with_obj_ids:
            if self.source_ids is None:
                raise KaraboSkyModelError(
                    "There are no 'source_ids' available in `sources`."
                )
            return np.hstack(
                (
                    self.sources.to_numpy(),
                    self.source_ids[self._sources_dim_sources]
                    .to_numpy()
                    .reshape(-1, 1),
                )
            )
        else:
            return self.sources.to_numpy()

    def rechunk_array_based_on_self(self, array: xr.DataArray) -> xr.DataArray:
        if self.sources is None:
            raise KaraboSkyModelError("Rechunking of `sources` None is not allowed.")
        if self.sources.chunks is not None:
            chunk_size = max(self.sources.chunks[0][0], 1)
            chunks: Dict[str, Any] = {self._sources_dim_sources: chunk_size}
            array = array.chunk(chunks=chunks)
        else:
            pass
        return array

    @overload
    def filter_by_radius(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[False] = False,
    ) -> SkyModel:
        ...

    @overload
    def filter_by_radius(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[True],
    ) -> Tuple[SkyModel, NDArray[np.int_]]:
        ...

    def filter_by_radius(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: bool = False,
    ) -> Union[SkyModel, Tuple[SkyModel, NDArray[np.int_]]]:
        """
        Filters the sky according to an inner and outer circle from the phase center

        Args:
            inner_radius_deg: Inner radius in degrees
            outer_radius_deg: Outer radius in degrees
            ra0_deg: Phase center right ascension
            dec0_deg: Phase center declination
            indices: Optional parameter, if set to True,
                we also return the indices of the filtered sky copy

        Returns:
            SkyModel: Filtered copy of the sky
        """
        copied_sky = SkyModel.copy_sky(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_radius`."
            )
        inner_circle = SphericalCircle(
            (ra0_deg * u.deg, dec0_deg * u.deg),
            inner_radius_deg * u.deg,
        )
        outer_circle = SphericalCircle(
            (ra0_deg * u.deg, dec0_deg * u.deg),
            outer_radius_deg * u.deg,
        )
        outer_sources = outer_circle.contains_points(copied_sky[:, 0:2])
        inner_sources = inner_circle.contains_points(copied_sky[:, 0:2])
        filtered_sources = np.logical_and(outer_sources, np.logical_not(inner_sources))
        filtered_sources_idxs = np.where(filtered_sources == True)[0]  # noqa: E712
        copied_sky.sources = copied_sky.sources[filtered_sources_idxs]

        # Rechunk the array to the original chunk size
        copied_sky.sources = self.rechunk_array_based_on_self(copied_sky.sources)

        if indices:
            return copied_sky, filtered_sources_idxs
        else:
            return copied_sky

    @overload
    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[False] = False,
    ) -> SkyModel:
        ...

    @overload
    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[True],
    ) -> Tuple[SkyModel, NDArray[np.int_]]:
        ...

    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: bool = False,
    ) -> Union[SkyModel, Tuple[SkyModel, NDArray[np.int_]]]:
        """
        Filters sources within an annular region using a flat Euclidean distance
        approximation suitable for large datasets managed by Xarray.

        This function is designed for scenarios where the dataset size precludes
        in-memory spherical geometry calculations. By leveraging a flat Euclidean
        approximation, it permits the use of Xarray's out-of-core computation
        capabilities, thus bypassing the limitations imposed by the incompatibility
        of `astropy.visualization.wcsaxes.SphericalCircle` with Xarray's data
        structures. Although this method trades off geometric accuracy against
        computational efficiency, it remains a practical choice for large angular
        fields where the curvature of the celestial sphere can be reasonably
        neglected.

        Args:
            inner_radius_deg : IntFloat
                The inner radius of the annular search region, in degrees.
            outer_radius_deg : IntFloat
                The outer radius of the annular search region, in degrees.
            ra0_deg : IntFloat
                The right ascension of the search region's center, in degrees.
            dec0_deg : IntFloat
                The declination of the search region's center, in degrees.
            indices : bool, optional
                If True, returns the indices of the filtered sources in addition to the
                SkyModel object. Defaults to False.

        Returns:
            SkyModel or tuple of (SkyModel, NDArray[np.int_]):
                The filtered SkyModel object, and optionally the indices of the filtered
                sources if `indices` is set to True.

        Raises:
            KaraboSkyModelError: If the `sources` attribute is not
            populated in the SkyModel instance prior to invoking this function.

        Note:
            Use this function for large sky models where a full spherical geometry
            calculation is not feasible due to memory constraints. It is particularly
            beneficial when working with Xarray and Dask, facilitating scalable data
            analysis on datasets that are too large to fit into memory.
        """
        copied_sky = SkyModel.copy_sky(self)

        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_radius`."
            )

        # Calculate distances to phase center using flat Euclidean approximation
        x = (copied_sky[:, 0] - ra0_deg) * np.cos(np.radians(dec0_deg))
        y = copied_sky[:, 1] - dec0_deg
        distances_sq = np.add(np.square(x), np.square(y))

        # Filter sources based on inner and outer radius
        filter_mask = cast(  # distances_sq actually an xr.DataArray because x & y are
            xr.DataArray,
            (distances_sq >= np.square(inner_radius_deg))
            & (distances_sq <= np.square(outer_radius_deg)),
        ).compute()

        copied_sky.sources = copied_sky.sources[filter_mask]

        copied_sky.sources = self.rechunk_array_based_on_self(copied_sky.sources)

        if indices:
            filtered_indices = np.where(filter_mask)[0]
            return copied_sky, filtered_indices
        else:
            return copied_sky

    def filter_by_column(
        self,
        col_idx: int,
        min_val: IntFloat,
        max_val: IntFloat,
    ) -> SkyModel:
        """
        Filters the sky based on a specific column index

        Args:
            col_idx: Column index to filter by
            min_val: Minimum value for the column
            max_val: Maximum value for the column

        Returns:
            SkyModel: Filtered copy of the sky
        """
        copied_sky = SkyModel.copy_sky(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before filtering."
            )

        # Create mask
        filter_mask = (copied_sky.sources[:, col_idx] >= min_val) & (
            copied_sky.sources[:, col_idx] <= max_val
        )
        filter_mask = self.rechunk_array_based_on_self(filter_mask).compute()

        # Apply the filter mask and drop the unmatched rows
        copied_sky.sources = copied_sky.sources.where(filter_mask, drop=True)

        return copied_sky

    def filter_by_flux(
        self,
        min_flux_jy: IntFloat,
        max_flux_jy: IntFloat,
    ) -> SkyModel:
        return self.filter_by_column(2, min_flux_jy, max_flux_jy)

    def filter_by_frequency(
        self,
        min_freq: IntFloat,
        max_freq: IntFloat,
    ) -> SkyModel:
        return self.filter_by_column(6, min_freq, max_freq)

    def get_wcs(self) -> WCS:
        """
        Gets the currently active world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html

        Returns:
            WCS: world coordinate system
        """
        return self.wcs

    def set_wcs(self, wcs: WCS) -> None:
        """
        Sets a new world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html

        Args:
            wcs: world coordinate system
        """
        self.wcs = wcs

    def setup_default_wcs(
        self,
        phase_center: IntFloatList = [0, 0],
    ) -> WCS:
        """
        Defines a default world coordinate system astropy.wcs
        For more details see https://docs.astropy.org/en/stable/wcs/index.html

        Args:
            phase_center: ra-dec location

        Returns:
            WCS: The new world coordinate system
        """
        w = WCS(naxis=2)
        w.wcs.crpix = [0, 0]  # coordinate reference pixel per axis
        w.wcs.cdelt = [-1, 1]  # coordinate increments on sphere per axis
        w.wcs.crval = phase_center
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type
        self.wcs = w
        return w

    @staticmethod
    def get_fits_catalog(path: str) -> Table:
        """
        Gets astropy.table.table.Table from a .fits catalog

        Args:
            path: Location of the .fits file

        Returns:
            astropy.table.table.Table: Table derived from fits catalog
        """
        return Table.read(path)

    def explore_sky(
        self,
        phase_center: IntFloatList,
        stokes: StokesType = "Stokes I",
        idx_to_plot: Optional[NDArray[np.int_]] = None,
        xlim: Optional[Tuple[IntFloat, IntFloat]] = None,
        ylim: Optional[Tuple[IntFloat, IntFloat]] = None,
        figsize: Optional[Tuple[IntFloat, IntFloat]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cfun: Optional[Callable[..., NPFloatInpBroadType]] = np.log10,
        cmap: Optional[str] = "plasma",
        cbar_label: Optional[str] = None,
        with_labels: bool = False,
        wcs: Optional[WCS] = None,
        wcs_enabled: bool = True,
        filename: Optional[str] = None,
        block: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        A scatter plot of the `SkyModel` (self) where the sources
        are projected according to the `phase_center`

        Args:
            phase_center: [RA,DEC]
            stokes: `SkyModel` stoke flux
            idx_to_plot: If you want to plot only a subset of the sources, set
                the indices here.
            xlim: RA-limit of plot
            ylim: DEC-limit of plot
            figsize: figsize as tuple
            title: plot title
            xlabel: x-axis label
            ylabel: y-axis label
            cfun: flux scale transformation function for scatter-coloring.
                Defaults to np.log10
            cmap: matplotlib color map
            cbar_label: color bar label
            with_labels: Plots object ID's if set?
            wcs: If you want to use a custom astropy.wcs, ignores `phase_center` if
                set
            wcs_enabled: Use wcs transformation?
            filename: Set to path/fname to save figure (set extension to fname to
                overwrite .png default)
            block: Whether or not plotting should block the rest of the program
            kwargs: matplotlib kwargs for scatter & Collections, e.g. customize `s`,
                `vmin` or `vmax`
        """
        # To avoid having to read the data multiple times, we read it once here
        if self.sources is None:
            raise KaraboSkyModelError("Can't plot sky if `sources` is None.")
        if idx_to_plot is not None:
            data = self.sources[idx_to_plot].as_numpy()
        else:
            data = self.sources.as_numpy()
        if wcs_enabled:
            if wcs is None:
                wcs = self.setup_default_wcs(phase_center)
            px, py = wcs.wcs_world2pix(
                self[:, 0], self[:, 1], 0
            )  # ra-dec transformation
            xlim_reset, ylim_reset = False, False
            if xlim is None and ylim is not None:
                xlim = (-1, 1)
                xlim_reset = True
            elif xlim is not None and ylim is None:
                ylim = (-1, 1)
                ylim_reset = True
            if xlim is not None and ylim is not None:
                xlim, ylim = wcs.wcs_world2pix(xlim, ylim, 0)
            if xlim_reset:
                xlim = None
            if ylim_reset:
                ylim = None
        else:
            px, py = self[:, 0], self[:, 1]

        flux = None
        if cmap is not None:
            flux = self[:, SkyModel._STOKES_IDX[stokes]].to_numpy()
            if cfun is not None:
                if cfun in [np.log10, np.log] and any(flux <= 0):
                    warn(
                        KaraboWarning(
                            "Warning: flux with value <= 0 found, setting "
                            "those to np.nan to avoid "
                            "logarithmic errors (only affects the colorbar)"
                        )
                    )

                    flux = np.where(flux > 0, flux, np.nan)
                flux = cast(NDArray[np.float_], cfun(flux))

        # handle matplotlib kwargs
        # not set as normal args because default assignment depends on args
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.nanmin(flux)  # type: ignore [arg-type]
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.nanmax(flux)  # type: ignore [arg-type]

        if wcs_enabled:
            slices = get_slices(wcs)
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection=wcs, slices=slices)
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(px, py, c=flux, cmap=cmap, **kwargs)

        if with_labels:
            unique_keys, indices = np.unique(
                data[self._sources_dim_sources], return_index=True
            )
            for i, txt in enumerate(unique_keys):
                if self.shape[0] > 1:
                    ax.annotate(
                        txt,
                        (px[indices][i], py[indices][i]),
                    )
                else:
                    ax.annotate(txt, (px, py))
        # Add grid
        ax.grid()
        plt.axis("equal")
        if cbar_label is None:
            cbar_label = ""
        plt.colorbar(sc, label=cbar_label)
        if title is not None:
            plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show(block=block)
        plt.pause(1)

        if isinstance(filename, str):
            fig.savefig(fname=filename)

    @staticmethod
    def get_OSKAR_sky(
        sky: Union[_SkySourcesType, SkyModel],
        precision: PrecisionType = "double",
    ) -> oskar.Sky:
        """
        Get OSKAR sky model object from the defined Sky Model

        Returns:
            OSKAR sky model
        """
        if sky.shape[1] > 12:
            return oskar.Sky.from_array(sky[:, :12], precision)
        else:
            return oskar.Sky.from_array(sky, precision)

    @staticmethod
    def read_healpix_file_to_sky_model_array(
        file: str,
        channel: int,
        polarisation: Polarisation,
    ) -> Tuple[xr.DataArray, int]:
        """
        Read a healpix file in hdf5 format.
        The file should have the map keywords:

        Args:
            file: hdf5 file path (healpix format)
            channel: Channels of observation (between 0 and maximum numbers of
                channels of observation)
            polarisation: 0 = Stokes I, 1 = Stokes Q, 2 = Stokes U, 3 = Stokes  V

        Returns:
            xr.DataArray: Array converted from healpix to 2-D array of RADEC
        """
        arr = get_healpix_image(file)
        filtered = arr[channel][polarisation.value]
        ra, dec, nside = convert_healpix_2_radec(filtered)
        return xr.DataArray(np.vstack((ra, dec, filtered)).transpose()), nside

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.sources is None:
            raise AttributeError(
                "`sources` is None and therefore has no `shape` attribute."
            )
        return self.sources.shape

    @property
    def num_sources(self) -> int:
        if self.sources is None:
            return 0
        else:
            return self.shape[0]

    @property
    def sources(self) -> Optional[xr.DataArray]:
        return self._sources

    @sources.setter
    def sources(self, value: Optional[_SkySourcesType]) -> None:
        """Sources setter.

        Does also allow numpy-arrays. But mypy doesn't allow getter and setter to
        have different dtypes (issue 3004). Just set "# type: ignore [assignment]"
        in case you don't have exactly an `xarray`.

        Args:
            value: sources, `xarray.DataArray` or `np.ndarray`
        """
        self._sources = None
        self._sources_dim_sources = XARRAY_DIM_0_DEFAULT
        self._sources_dim_data = XARRAY_DIM_1_DEFAULT
        if value is not None:
            self.add_point_sources(sources=value)

    @property
    def _sources_dim_sources(self) -> str:
        if self.sources is None:
            return XARRAY_DIM_0_DEFAULT
        else:
            return self.__sources_dim_sources

    @_sources_dim_sources.setter
    def _sources_dim_sources(self, value: str) -> None:
        if (
            self._sources_dim_sources != XARRAY_DIM_0_DEFAULT
            and self._sources_dim_sources != value
        ):
            raise KaraboSkyModelError(
                "Provided dim_0 name is not consistent with existing dim-name of "
                + f"`sources`. Provided dim-name is {value} but existing is "
                + f"{self._sources_dim_sources}."
            )
        self.__sources_dim_sources = value

    @property
    def _sources_dim_data(self) -> str:
        if self.sources is None:
            return XARRAY_DIM_1_DEFAULT
        else:
            return self.__sources_dim_data

    @_sources_dim_data.setter
    def _sources_dim_data(self, value: str) -> None:
        if (
            self._sources_dim_data != XARRAY_DIM_1_DEFAULT
            and self._sources_dim_data != value
        ):
            raise KaraboSkyModelError(
                "Provided dim_1 name is not consistent with existing dim-name of "
                + f"`sources`. Provided dim-name is {value} but existing is "
                + f"{self._sources_dim_data}."
            )
        self.__sources_dim_data = value

    @property
    def source_ids(self) -> Optional[DataArrayCoordinates[xr.DataArray]]:
        if self.sources is not None and len(self.sources.coords) > 0:
            return self.sources.coords
        else:
            return None

    @source_ids.setter
    def source_ids(
        self,
        value: Optional[_SourceIdType],
    ) -> None:
        if self.sources is None or self._sources is None:
            raise KaraboSkyModelError(
                "Setting source-ids on empty `sources` is not allowed."
            )
        if value is None:
            if self._sources_dim_sources in self._sources.indexes:
                self._sources = self._sources.reset_index(
                    self._sources_dim_sources, drop=True
                )
        else:
            if isinstance(value, DataArrayCoordinates):
                if self._sources_dim_sources not in value:
                    raise KaraboSkyModelError(
                        f"Coord key {self._sources_dim_sources} not found in "
                        + "provided `DataArrayCoordinates`."
                    )
                n_provided = value[self._sources_dim_sources].shape[0]
            else:
                n_provided = len(value)
            if self.sources.shape[0] != n_provided:
                raise KaraboSkyModelError(
                    f"Number of provided sources of `source_ids` {n_provided} does "
                    f"not match the number of existing sources {self.sources.shape[0]}."
                )
            self.sources.coords[self._sources_dim_sources] = value

    def __getitem__(self, key: Any) -> xr.DataArray:
        """
        Allows to get access to self.sources in an np.ndarray like manner
        If casts the selected array/scalar to float64 if possible
        (usually if source_id is not selected)

        Args:
            key: slice key

        Returns:
            sliced self.sources
        """
        if self.sources is None:
            raise AttributeError("`sources` is None and therefore can't be accessed.")
        return self.sources[key]

    def __setitem__(
        self,
        key: Any,
        value: Union[_SkySourcesType, NPFloatLike],
    ) -> None:
        """
        Allows to set values in an np.ndarray like manner

        Args:
            key: slice key
            value: values to store
        """
        if self.sources is None:
            raise KaraboSkyModelError("Can't access `sources` because it's None.")
        # access `sources.getter`, not `sources.setter` which is fine
        self.sources[key] = value

    def save_sky_model_as_csv(self, path: str) -> None:
        """
        Save source array into a csv.

        Args:
            path: path to save the csv file in.
        """
        if self.sources is None:
            raise KaraboSkyModelError("Can't save `sources` because they're None.")
        df = pd.DataFrame(self.sources)
        df["source id (object)"] = self.sources[self._sources_dim_sources].values
        df.to_csv(
            path,
            index=False,
            header=[
                "right ascension (deg)",
                "declination (deg)",
                "stokes I Flux (Jy)",
                "stokes Q Flux (Jy)",
                "stokes U Flux (Jy)",
                "stokes V Flux (Jy)",
                "reference_frequency (Hz)",
                "spectral index (N/A)",
                "rotation measure (rad / m^2)",
                "major axis FWHM (arcsec)",
                "minor axis FWHM (arcsec)",
                "position angle (deg)",
                "true redshift",
                "observed redshift",
                "source id (object)",
            ],
        )

    @staticmethod
    def __convert_ra_dec_to_cartesian(
        ra: IntFloat, dec: IntFloat
    ) -> NDArray[np.float_]:
        x = math.cos(math.radians(ra)) * math.cos(math.radians(dec))
        y = math.sin(math.radians(ra)) * math.cos(math.radians(dec))
        z = math.sin(math.radians(dec))
        r = np.array([x, y, z])
        norm = cast(np.float_, np.linalg.norm(r))
        if norm == 0:
            return r
        return r / norm

    def get_cartesian_sky(self) -> NDArray[np.float_]:
        if self.sources is None:
            raise AttributeError("Can't create cartesian-sky when `sources` is None.")
        cartesian_sky = np.squeeze(
            np.apply_along_axis(
                lambda row: [
                    self.__convert_ra_dec_to_cartesian(float(row[0]), float(row[1]))
                ],
                axis=1,
                arr=self.sources,
            )
        )
        return cartesian_sky

    @classmethod
    def get_sky_model_from_h5_to_xarray(
        cls: Type[_TSkyModel],
        path: str,
        prefix_mapping: Optional[SkyPrefixMapping] = None,
        load_as: Literal["numpy_array", "dask_array"] = "dask_array",
        chunksize: Union[int, Literal["auto"]] = "auto",
    ) -> _TSkyModel:
        """
        Load a sky model dataset from an HDF5 file and
        converts it to an xarray DataArray.

        Args:
            path : Path to the input HDF5 file.
            prefix_mapping : Mapping column names to their
                corresponding dataset paths in the HDF5 file.
                If the column is not present in the HDF5 file,
                set its value to None.
            load_as : Literal["numpy_array", "dask_array"], default="dask_array"
                What type of array to load the data inside the xarray Data Array as.
            chunksize : Union[int, str], default=auto
                Chunk size for Dask arrays. This determines the size of chunks that
                the data will be divided into when read from the file. Can be an
                integer or 'auto'. If 'auto', Dask will choose an optimal chunk size.

        Returns:
            xr.DataArray: A 2D xarray DataArray containing the sky model data.
            Rows represent data points and columns represent
            different data fields ('ra', 'dec', ...).
        """
        f = h5py.File(path, "r", locking=False)
        data_arrays: List[xr.DataArray] = []

        # not sure why we have a default here, but keep for compatibility I guess?
        if prefix_mapping is None:
            prefix_mapping = SkyPrefixMapping(
                ra="Right Ascension",
                dec="Declination",
                stokes_i="Flux",
                observed_redshift="Observed Redshift",
            )

        for field in fields(prefix_mapping):
            field_value: Optional[str] = getattr(prefix_mapping, field.name)
            if field_value is None:
                shape = f[prefix_mapping.ra].shape
                dask_array = da.zeros(shape, chunks=(chunksize,))  # type: ignore [attr-defined] # noqa: E501
            else:
                dask_array = da.from_array(f[field_value], chunks=(chunksize,))  # type: ignore [attr-defined] # noqa: E501
            data_arrays.append(xr.DataArray(dask_array, dims=[XARRAY_DIM_0_DEFAULT]))

        if load_as == "numpy_array":
            data_arrays = [x.compute() for x in data_arrays]
        sky = xr.concat(data_arrays, dim=XARRAY_DIM_1_DEFAULT)
        sky = sky.T
        chunks: Dict[str, Any] = {
            XARRAY_DIM_0_DEFAULT: chunksize,
            XARRAY_DIM_1_DEFAULT: sky.shape[1],
        }
        sky = sky.chunk(chunks=chunks)
        return cls(sky, h5_file_connection=f)

    @classmethod
    def _get_sky_model_from_freq_formatted_fits(
        cls: Type[_TSkyModel],
        *,
        data: FITS_rec,
        cols: ColDefs,
        prefix_mapping: SkyPrefixMapping,
        min_freq: Optional[float],
        max_freq: Optional[float],
        unit_mapping: Dict[str, UnitBase],
        units_sources: SkySourcesUnits,
        chunks_fun: Callable[[xr.DataArray], xr.DataArray],
        zero_col_array_fun: Callable[[int], xr.DataArray],
        encoded_freq: UnitBase,
    ) -> _TSkyModel:
        """Creates a sky-model from a .fits file, where the frequency is encoded
        in the according col-names.

        Args:
            data: Data of .fits file.
            cols: Columns of .fits file.
            prefix_mapping: Formattable col-names of .fits file.
            min_freq: Filter by min-freq in Hz? May increase file-reading significantly.
            max_freq: Filter by max-freq in Hz? May increase file-reading significantly.
            unit_mapping: Mapping from col-unit to `astropy.unit`.
            units_sources: Units of `SkyModel.sources`.
            chunks_fun: Callback to handle array-chunking.
            zero_col_array_fun: Callback to create a (`length`,1) zero `DataArray`.
            encoded_freq: Unit of col-name encoded frequency.

        Returns:
            SkyModel: Sky model with according sources from `data`.
        """
        if prefix_mapping.ref_freq is not None:
            raise RuntimeError(
                "Setting `prefix_mapping.ref_freq` and `encoded_freq` is "
                + "ambiguous and therefore not allowed."
            )
        (
            prefix_mapping,
            num_formatted,
            names_and_freqs,  # here, freqs are already in Hz
        ) = SkySourcesUnits.format_sky_prefix_freq_mapping(
            cols=cols,
            prefix_mapping=prefix_mapping,
            encoded_freq=encoded_freq,
        )
        if num_formatted <= 0:
            raise RuntimeError(_DEV_ERROR_MSG)
        unit_scales = units_sources.get_unit_scales(
            cols=cols,
            unit_mapping=unit_mapping,
            prefix_mapping=prefix_mapping,
        )
        cache: Dict[str, xr.DataArray] = dict()
        sources: Optional[xr.DataArray] = None
        col_data: xr.DataArray
        for i in range(num_formatted):
            freq: Optional[float] = None
            sources_i: Optional[xr.DataArray] = None
            broken = False
            for spm_field in fields(prefix_mapping):
                if (
                    field_values := getattr(prefix_mapping, spm_field.name)
                ) is not None:
                    # check if it's a formatted field or not
                    if not isinstance(field_values, list):
                        if field_values not in cache.keys():
                            col_data = xr.DataArray(data=data[field_values]).T
                            if field_values in unit_scales.keys():
                                col_data *= unit_scales[field_values]
                            col_data = chunks_fun(col_data)
                            cache[field_values] = col_data
                        else:
                            col_data = cache[field_values]
                    else:  # here, we deal with a formatted SkyPrefixMapping field
                        col_name: str = field_values[i]
                        col_freq = names_and_freqs[col_name]
                        if (min_freq is not None and col_freq < min_freq) or (
                            max_freq is not None and col_freq > max_freq
                        ):  # freq-filtering in case of formatted `prefix_mapping`
                            broken = True
                            break
                        if freq is None:
                            freq = col_freq
                        else:
                            if col_freq != freq:
                                raise RuntimeError(
                                    "Order of formatted sky-prefixes is not right! "
                                    + "This is probably an error of Karabo. Please "
                                    + "check `SkyPrefixMapping` list-order: "
                                    + f"{prefix_mapping}"
                                )
                        col_data = xr.DataArray(data=data[col_name]).T
                        if col_name in unit_scales.keys():
                            col_data *= unit_scales[col_name]
                        col_data = chunks_fun(col_data)
                    # create xarray for current format
                    if sources_i is None:
                        sources_i = xr.concat(
                            (
                                zero_col_array_fun(col_data.shape[0])
                                for _ in range(cls.SOURCES_COLS - 1)
                            ),
                            dim=XARRAY_DIM_1_DEFAULT,
                        )
                        sources_i = chunks_fun(sources_i)
                    if (field_name := spm_field.name) == "id":
                        sources_i.coords[XARRAY_DIM_0_DEFAULT] = col_data
                    else:
                        col_idx = cls.COL_IDX[field_name]  # type: ignore[index]
                        sources_i[:, col_idx] = col_data
            if broken:
                continue
            if sources_i is None:  # num_formatted <= 0 should be impossible
                raise RuntimeError(_DEV_ERROR_MSG)
            if freq is None:
                raise RuntimeError(_DEV_ERROR_MSG)
            freqs = xr.DataArray(data=np.repeat(freq, repeats=col_data.shape[0])).T
            freqs = chunks_fun(freqs)
            sources_i[:, cls.COL_IDX["ref_freq"]] = freqs
            if sources is None:
                sources = sources_i
            else:
                sources = xr.concat((sources, sources_i), dim=XARRAY_DIM_0_DEFAULT)
                sources = chunks_fun(sources)

        if sources is None:
            raise ValueError(_DEV_ERROR_MSG)

        return cls(sources=sources)

    @classmethod
    def _get_sky_model_from_non_formatted_fits(
        cls: Type[_TSkyModel],
        *,
        data: FITS_rec,
        cols: ColDefs,
        prefix_mapping: SkyPrefixMapping,
        min_freq: Optional[float],
        max_freq: Optional[float],
        unit_mapping: Dict[str, UnitBase],
        units_sources: SkySourcesUnits,
        chunks_fun: Callable[[xr.DataArray], xr.DataArray],
        zero_col_array_fun: Callable[[int], xr.DataArray],
    ) -> _TSkyModel:
        """Creates a sky-model from a .fits file, where `SkyModel.sources` come from
        distinct data-arrays, not from a combination.

        Args:
            data: Data of .fits file.
            cols: Columns of .fits file.
            prefix_mapping: NON-formattable col-names of .fits file.
            min_freq: Filter by min-freq in Hz? May increase file-reading significantly.
            max_freq: Filter by max-freq in Hz? May increase file-reading significantly.
            unit_mapping: Mapping from col-unit to `astropy.unit`.
            units_sources: Units of `SkyModel.sources`.
            chunks_fun: Callback to handle array-chunking.
            zero_col_array_fun: Callback to create a (`length`,1) zero `DataArray`.

        Returns:
            sky-model with according sources from `data`.
        """
        unit_scales = units_sources.get_unit_scales(
            cols=cols,
            unit_mapping=unit_mapping,
            prefix_mapping=prefix_mapping,
        )
        if (
            min_freq is not None or max_freq is not None
        ) and prefix_mapping.ref_freq is None:
            raise RuntimeError(
                f"`prefix_mapping.ref_freq` has to exist if {min_freq=} and/or "
                + f"{max_freq=} is set."
            )
        freqs: Optional[xr.DataArray] = None
        freq_idxs: Optional[xr.DataArray] = None
        sources: Optional[xr.DataArray] = None
        if (ref_freq_col_name := prefix_mapping.ref_freq) is not None:
            # do freq-filtering of non-formatted `prefix_mapping` here to
            # determine number-of-sources at the start
            freqs = (
                xr.DataArray(data=data[ref_freq_col_name]).T
                * unit_scales[ref_freq_col_name]
            )
            if min_freq is not None and max_freq is not None:
                freq_idxs = cast(  # cast to match run-time type
                    xr.DataArray, np.logical_and(freqs < max_freq, freqs > min_freq)
                )
            else:
                if min_freq is not None:
                    freq_idxs = freqs > min_freq
                if max_freq is not None:
                    freq_idxs = freqs < max_freq
                if freq_idxs is not None and all(freq_idxs):
                    freq_idxs = None  # no need for filtering if freq don't constrain
            if freq_idxs is not None:
                freqs = freqs[freq_idxs]
        for spm_field in fields(prefix_mapping):
            if (field_value := getattr(prefix_mapping, spm_field.name)) is not None:
                if isinstance(field_value, list):  # no formatted field-values allowed
                    raise RuntimeError(_DEV_ERROR_MSG)
                col_data = xr.DataArray(data=data[field_value]).T
                if freq_idxs is not None:
                    col_data = col_data[freq_idxs]
                if field_value in unit_scales.keys():
                    col_data *= unit_scales[field_value]
                col_data = chunks_fun(col_data)

                if sources is None:
                    sources = xr.concat(
                        (
                            zero_col_array_fun(col_data.shape[0])
                            for _ in range(cls.SOURCES_COLS - 1)
                        ),
                        dim=XARRAY_DIM_1_DEFAULT,
                    )
                    sources = chunks_fun(sources)
                if (field_name := spm_field.name) == "id":
                    sources.coords[XARRAY_DIM_0_DEFAULT] = col_data
                else:
                    col_idx = cls.COL_IDX[field_name]  # type: ignore[index]
                    sources[:, col_idx] = col_data
        if sources is None:  # `prefix_mapping` having no fields doesn't make any sense
            raise RuntimeError(_DEV_ERROR_MSG)
        return cls(sources=sources)

    @classmethod
    def get_sky_model_from_fits(
        cls: Type[_TSkyModel],
        *,
        fits_file: FilePathType,
        prefix_mapping: SkyPrefixMapping,
        unit_mapping: Dict[str, UnitBase],
        units_sources: Optional[SkySourcesUnits] = None,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        encoded_freq: Optional[UnitBase] = None,
        chunks: Optional[_ChunksType] = "auto",
        memmap: bool = False,
    ) -> _TSkyModel:
        """
        Creates a sky-model from `fits_file`. The following formats are supported:

        - Each data-array of the .fits file maps to a single `SkyModel.sources` \
        column.

        - Frequency of some columns is encoded in col-names of the .fits file.

        Args:
            fits_file: The .fits file to create the sky-model from.
            prefix_mapping: Formattable col-names of .fits file. If `encoded_freq`
                is not None, the freq-encoded field-values must have '{0}'
                as placeholder.
            unit_mapping: Mapping from col-unit to `astropy.unit`.
            units_sources: Units of `SkyModel.sources`.
            min_freq: Filter by min-freq in Hz. May slow down reading of
                file significantly.
            max_freq: Filter by max-freq in Hz. May slow down reading of
                file significantly.
            encoded_freq: Unit of col-name encoded frequency, if the .fits file has
                it's ref-frequency encoded in the col-names.
            chunks: Coerce the array's data into dask arrays with the given chunks.
                Might be useful for reading larger-than-memory .fits files.
            memmap: Whether to use memory mapping when opening the FITS file.
                Allows for reading of larger-than-memory files.

        Returns:
            SkyModel: Sky model with according sources from `fits_file`.
        """

        def chunks_fun(arr: xr.DataArray) -> xr.DataArray:
            if chunks is not None:
                arr = arr.chunk(chunks=chunks)
            return arr

        def get_zero_col_array(length: int) -> xr.DataArray:
            col_array = xr.DataArray(data=np.zeros(shape=(length, 1)))
            return chunks_fun(col_array)

        if units_sources is None:  # load default dataclass if not specified
            units_sources = SkySourcesUnits()
        with fits.open(fits_file, memmap=memmap) as hdul:
            data: FITS_rec = hdul[1].data
            cols: ColDefs = hdul[1].columns

        if encoded_freq is None:
            sky = cls._get_sky_model_from_non_formatted_fits(
                data=data,
                cols=cols,
                prefix_mapping=prefix_mapping,
                min_freq=min_freq,
                max_freq=max_freq,
                unit_mapping=unit_mapping,
                units_sources=units_sources,
                chunks_fun=chunks_fun,
                zero_col_array_fun=get_zero_col_array,
            )
        else:
            sky = cls._get_sky_model_from_freq_formatted_fits(
                data=data,
                cols=cols,
                prefix_mapping=prefix_mapping,
                min_freq=min_freq,
                max_freq=max_freq,
                unit_mapping=unit_mapping,
                units_sources=units_sources,
                chunks_fun=chunks_fun,
                zero_col_array_fun=get_zero_col_array,
                encoded_freq=encoded_freq,
            )
        if sky.num_sources == 0:
            raise KaraboSkyModelError(
                "Selected Sky has not a single source. This is probably because "
                + "provided `min_freq` and/or `max_freq` are out of the survey's "
                + "frequencies and therefore results in filtering each source."
            )
        return sky

    @classmethod
    def get_GLEAM_Sky(
        cls: Type[_TSkyModel],
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
    ) -> _TSkyModel:
        """Creates a SkyModel containing GLEAM sources from
        https://vizier.cfa.harvard.edu/ service with more than 300,000 sources
        on different frequency-bands.

        The survey was created using MWA-telescope and covers the entire sky south
        of dec +30.

        GLEAM's frequency-range: 72-231 MHz

        The required .fits file will get downloaded and cached on disk.

        Args:
            min_freq: Set min-frequency in Hz for pre-filtering.
            max_freq: Set max-frequency in Hz for pre-filtering.

        Returns:
            GLEAM sky as `SkyModel`.
        """
        survey_file = GLEAMSurveyDownloadObject().get()
        encoded_freq = u.MHz
        unit_mapping = {
            "Jy": u.Jy,
            "Jy/beam": u.Jy / u.beam,
            "deg": u.deg,
            "arcsec": u.arcsec,
        }
        prefix_mapping = SkyPrefixMapping(
            ra="RAJ2000",
            dec="DEJ2000",
            stokes_i="Fp{0}",
            major="a{0}",
            minor="b{0}",
            pa="pa{0}",
            id="GLEAM",
        )
        units_sources = SkySourcesUnits(
            stokes_i=u.Jy / u.beam,
        )

        return cls.get_sky_model_from_fits(
            fits_file=survey_file,
            prefix_mapping=prefix_mapping,
            unit_mapping=unit_mapping,
            units_sources=units_sources,
            min_freq=min_freq,
            max_freq=max_freq,
            encoded_freq=encoded_freq,
        )

    @classmethod
    def get_sample_simulated_catalog(cls: Type[_TSkyModel]) -> _TSkyModel:
        """ Downloads a sample simulated HI source catalog and generates a sky
        model using the downloaded data. The catalog size is around 8MB.

        Source: The simulated catalog data was provided by Luis Machado
        (https://github.com/lmachadopolettivalle) in collaboration
        with the ETHZ Cosmology Research Group.

        Returns:
            SkyModel: The corresponding sky model.
            The sky model contains the following information:
                - 'Right Ascension' (ra): The right ascension coordinates \
                of the celestial objects.
                - 'Declination' (dec): The declination coordinates of the \
                celestial objects.
                - 'Flux' (i): The flux measurements of the celestial objects.
                - 'Observed Redshift': Additional observed redshift information \
                of the celestial objects.

        Note:
            Other properties such as 'stokes_q', 'stokes_u', 'stokes_v',
            'ref_freq', 'spectral_index', 'rm', 'major', 'minor', 'pa', and 'id'
            are not included in the sky model.
        """
        survey = HISourcesSmallCatalogDownloadObject()
        path = survey.get()
        sky = cls.get_sky_model_from_h5_to_xarray(path=path)
        if sky.sources is None:
            raise KaraboSkyModelError("`sky.sources` is None, which is unexpected.")

        return sky

    @classmethod
    def get_MIGHTEE_Sky(
        cls: Type[_TSkyModel],
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
    ) -> _TSkyModel:
        """Creates a SkyModel containing "MIGHTEE Continuum Early Science L1 catalogue"
        sources from https://archive.sarao.ac.za service, consisting of 9896 sources.

        MIGHTEE is an extragalactic radio-survey carried out using MeerKAT-telescope.
        It covers roughly the area between RA[149.40,150.84] and DEC[1.49,2.93].

        MIGHTEES's frequency-range: 1304-1375 MHz

        The required .fits file will get downloaded and cached on disk.

        Args:
            min_freq: Set min-frequency in Hz for pre-filtering.
            max_freq: Set max-frequency in Hz for pre-filtering.

        Returns:
            MIGHTEE sky as `SkyModel`.
        """
        survey_file = MIGHTEESurveyDownloadObject().get()
        unit_mapping: Dict[str, UnitBase] = {
            "DEG": u.deg,
            "JY": u.Jy,
            "JY/BEAM": u.Jy / u.beam,
            "HZ": u.Hz,
        }
        prefix_mapping = SkyPrefixMapping(
            ra="RA",
            dec="DEC",
            stokes_i="S_PEAK",
            ref_freq="NU_EFF",
            major="IM_MAJ",
            minor="IM_MIN",
            pa="IM_PA",
            id="NAME",
        )
        units_sources = SkySourcesUnits(
            stokes_i=u.Jy / u.beam,
        )
        return cls.get_sky_model_from_fits(
            fits_file=survey_file,
            prefix_mapping=prefix_mapping,
            unit_mapping=unit_mapping,
            units_sources=units_sources,
            min_freq=min_freq,
            max_freq=max_freq,
            encoded_freq=None,
            memmap=False,
        )

    @classmethod
    def get_MALS_DR1V3_Sky(
        cls: Type[_TSkyModel],
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
    ) -> _TSkyModel:
        """Creates a SkyModel containing "MALS V3" catalogue, of 715,760 sources,
        where the catalogue and it's information are from 'https://mals.iucaa.in/'.

        The MeerKAT Absorption Line Survey (MALS) consists of 1655 hrs of MeerKAT time
        (anticipated raw data ~ 1.7 PB) to carry out the most sensitive search of HI
        and OH absorption lines at 0<z<2, the redshift range over which most of the
        cosmic evolution in the star formation rate density takes place. The MALS
        survey is described in 'Gupta et al. (2016)'.

        MALS sky-coverage consists of 392 sources trackings between all RA-angles
        and DEC[-78.80,32.35] and different radius.

        MALS's frequency-range: 902-1644 MHz.

        For publications, please honor their work by citing them as follows:
        - If you describe MALS or associated science, please cite 'Gupta et al. 2016'.
        - If you use DR1 data products, please cite 'Deka et al. 2024'.

        Args:
            min_freq: Set min-frequency in Hz for pre-filtering.
            max_freq: Set max-frequency in Hz for pre-filtering.

        Returns:
            MALS sky as `SkyModel`.
        """
        survey_file = MALSSurveyV3DownloadObject().get()
        unit_mapping: dict[str, UnitBase] = {
            "deg": u.deg,
            "beam-1 mJy": u.mJy / u.beam,
            "MHz": u.MHz,
            "arcsec": u.arcsec,
        }
        prefix_mapping = SkyPrefixMapping(
            ra="ra_max",
            dec="dec_max",
            stokes_i="peak_flux",
            ref_freq="ref_freq",
            major="maj",
            minor="min",
            pa="pa",
            id="source_name",
        )
        unit_sources = SkySourcesUnits(
            stokes_i=u.Jy / u.beam,
        )
        return cls.get_sky_model_from_fits(
            fits_file=survey_file,
            prefix_mapping=prefix_mapping,
            unit_mapping=unit_mapping,
            units_sources=unit_sources,
            min_freq=min_freq,
            max_freq=max_freq,
        )

    @classmethod
    def get_random_poisson_disk_sky(
        cls: Type[_TSkyModel],
        min_size: Tuple[IntFloat, IntFloat],
        max_size: Tuple[IntFloat, IntFloat],
        flux_min: IntFloat,
        flux_max: IntFloat,
        r: int = 3,
    ) -> _TSkyModel:
        sky_array = xr.DataArray(
            get_poisson_disk_sky(min_size, max_size, flux_min, flux_max, r)
        )
        return cls(sky_array)

    @classmethod
    def sky_test(cls: Type[_TSkyModel]) -> _TSkyModel:
        """
        Construction of a sky model which can be used for testing and visualizing the
        simulation with equal distributed point sources around the phase center ra=20,
        deg=-30.

        Returns:
             SkyModel: The test sky model.
        """
        sky = cls()
        sky_data = np.zeros((81, cls.SOURCES_COLS))
        a = np.arange(-32, -27.5, 0.5)
        b = np.arange(18, 22.5, 0.5)
        dec_arr, ra_arr = np.meshgrid(a, b)
        sky_data[:, 0] = ra_arr.flatten()
        sky_data[:, 1] = dec_arr.flatten()
        sky_data[:, 2] = 1

        sky.add_point_sources(sky_data)

        return sky

    @overload
    def convert_to_backend(
        self,
        backend: Literal[SimulatorBackend.OSKAR] = SimulatorBackend.OSKAR,
        desired_frequencies_hz: Literal[None] = None,
        channel_bandwidth_hz: Optional[float] = None,
        verbose: bool = False,
    ) -> SkyModel:
        ...

    @overload
    def convert_to_backend(
        self,
        backend: Literal[SimulatorBackend.RASCIL],
        desired_frequencies_hz: NDArray[np.float_],
        channel_bandwidth_hz: Optional[float] = None,
        verbose: bool = False,
    ) -> List[SkyComponent]:
        ...

    def convert_to_backend(
        self,
        backend: SimulatorBackend = SimulatorBackend.OSKAR,
        desired_frequencies_hz: Optional[NDArray[np.float_]] = None,
        channel_bandwidth_hz: Optional[float] = None,
        verbose: bool = False,
    ) -> Union[SkyModel, List[SkyComponent]]:
        """Convert an existing SkyModel instance into a format acceptable
        by a desired backend.

        Args:
            backend: Determines how to return the SkyModel source catalog.

                    - OSKAR: return the current SkyModel instance, since methods \
                    in Karabo support OSKAR-formatted source np.array values.
                    - RASCIL: convert the current source array into a \
                    list of RASCIL SkyComponent instances.
            desired_frequencies_hz: List of frequencies corresponding to start
                of desired frequency channels. This field is required
                to convert sources into RASCIL SkyComponents.
                The array contains starting frequencies for the desired channels.
                E.g. [100e6, 110e6] corresponds to 2 frequency channels,
                which start at 100 MHz and 110 MHz, both with a bandwidth of 10 MHz.
                channel_bandwidth_hz: Used if desired_frequencies_hz has
                only one element. Otherwise, bandwidth is determined as
                the delta between the first two entries in desired_frequencies_hz.
            verbose: Determines whether to display additional print statements.

        Returns:
            SkyModel: A SkyModel suitable for your chosen backend, e.g. OSKAR
        """

        if backend is SimulatorBackend.OSKAR:
            if verbose is True:
                print(
                    """Desired backend is OSKAR.
                    Will not modify existing SkyModel instance."""
                )
            return self
        elif backend is SimulatorBackend.RASCIL:
            if verbose is True:
                print(
                    """Desired backend is RASCIL.
                    Will convert sources into a list of
                    RASCIL SkyComponent instances."""
                )

            desired_frequencies_hz = cast(NDArray[np.float_], desired_frequencies_hz)

            assert (
                len(desired_frequencies_hz) > 0
            ), """Must have at least 1 element
            in desired_frequencies_hz array"""

            desired_frequencies_hz = np.sort(desired_frequencies_hz)

            if len(desired_frequencies_hz) == 1:
                if channel_bandwidth_hz is None:
                    raise ValueError(
                        """desired_frequencies_hz has one entry
                        and channel_bandwidth_hz is None.
                        There is not enough information to find channel bandwidths.
                        Please specify channel_bandwidth_hz,
                        or add entries to desired_frequencies_hz."""
                    )
                frequency_bandwidth = channel_bandwidth_hz
            else:
                frequency_bandwidth = (
                    desired_frequencies_hz[1] - desired_frequencies_hz[0]
                )

            frequency_channel_centers = desired_frequencies_hz + frequency_bandwidth / 2

            # Set endpoints, i.e. all channel starts + the final channel's end
            frequency_channel_endpoints = np.append(
                desired_frequencies_hz, desired_frequencies_hz[-1] + frequency_bandwidth
            )

            # 1. Remove sources that fall outside all desired frequency channels
            # 2. Assign each source to the frequency channel closest
            # to its corresponding redshift (using np.digitize)
            # 3. For each source, create a SkyComponent,
            # with flux array equal to 0 on all channels,
            # except for its closest channel, where all its flux will belong
            # This is equivalent to having the source's SED equal to a delta function
            # at the frequency corresponding to its redshift,
            # which is true for line emission catalogues.
            redshift_limits = convert_frequency_to_z(
                np.array(
                    [
                        np.max(frequency_channel_endpoints),
                        np.min(frequency_channel_endpoints),
                    ]
                )
            )
            min_redshift, max_redshift = cast(
                Tuple[np.float_, np.float_], redshift_limits
            )

            if self.sources is None:
                return []

            assert self.sources is not None

            redshift_mask = (self.sources[:, 13] <= max_redshift) & (
                self.sources[:, 13] >= min_redshift
            )
            if verbose is True:
                print(min_redshift, max_redshift)
                print(self.sources)
            ras = self.sources[:, 0][redshift_mask]  # Degrees
            decs = self.sources[:, 1][redshift_mask]  # Degrees
            fluxes = self.sources[:, 2][redshift_mask]  # Jy * MHz
            redshifts = self.sources[:, 13][redshift_mask]
            if verbose is True:
                print(
                    f"""Reduced size of source catalog, after removing sources
                    outside of desired frequency range: {redshifts.shape}"""
                )

            # For each source, find the channel to which it belongs
            source_channel_indices = np.digitize(
                convert_z_to_frequency(redshifts),
                frequency_channel_endpoints,
                right=False,
            )

            # E.g. if channel starts are [1e8, 2e8],
            # then a source at frequency 1.5e8 should fall into the 0th channel.
            # However, digitize returns index 1 for such a source.
            # Therefore, we subtract 1 from the return value of np.digitize
            source_channel_indices -= 1

            skycomponents: List[SkyComponent] = []
            for ra, dec, flux, index in zip(
                ras,
                decs,
                fluxes,
                source_channel_indices,
            ):
                # 1 == npolarisations, fixed as 1 (stokesI) for now
                # TODO eventually handle full stokes source catalogs
                flux_array = np.zeros((len(frequency_channel_endpoints) - 1, 1))

                # Access [0] since this is the stokesI flux,
                # and [index] to place the source's flux onto
                # the correct frequency channel (since this works for line emission)
                flux_array[index][0] = flux

                skycomponents.append(
                    SkyComponent(
                        direction=SkyCoord(
                            ra=ra,
                            dec=dec,
                            unit="deg",
                            frame="icrs",
                            equinox="J2000",
                        ),
                        frequency=frequency_channel_centers,
                        name=f"pointsource{ra}{dec}",
                        flux=flux_array,  # shape: nchannels, npolarisations
                        shape="Point",
                        polarisation_frame=PolarisationFrame("stokesI"),
                        params=None,
                    )
                )
            return skycomponents

        assert_never(backend)
