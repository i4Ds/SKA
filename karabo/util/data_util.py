"""
This module offers a variety of functions connected to string manipulation,
file paths and data handling.
"""
import os
import re
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

import karabo
from karabo.util._types import NPIntFloat
from karabo.util.math_util import Voigt


def get_module_absolute_path() -> str:
    """
    Returns the absolute path to the Karabo module.
    This function is mainly used to find the data directory of the package.
    It has the telescope setup files.


    Returns:
        The absolute path to the Karabo module.
    """
    path_elements = os.path.abspath(karabo.__file__).split(os.path.sep)
    path_elements.pop()
    return os.path.sep.join(path_elements)


def get_module_path_of_module(module: ModuleType) -> str:
    """
    Returns the absolute path to a module in Karabo package.
    It is used to find data or sample directories in third-party packages.

    Args:
        module (ModuleType): The name of the module.

    Returns:
        The absolute path to the Karabo module.
    """
    module_file = cast(str, module.__file__)
    path_elements = os.path.abspath(module_file).split(os.path.sep)
    path_elements.pop()
    return os.path.sep.join(path_elements)


def extract_digit_from_string(string: str) -> int:
    """
    Extracts the digits from an alphanumeric string. Example:

    "abc123efg666gggg" --> 123666

    Args:
        string (str): The string to be parsed

    Returns:
        int: The number made from the digits.
    """
    digit = ""
    for char in string:
        if char.isdigit():
            digit += char
    return int(digit)


def extract_chars_from_string(string: str) -> str:
    """
    Removes all digits from an alphanumeric string. Example:

    "abc123efg666gggg" --> "abcefggggg"

    Args:
        string (str): The string to be parsed

    Returns:
        str: A string with all digits removed.
    """
    letters = ""
    for char in string:
        if char.isalpha():
            letters += char
    return letters


def parse_size(size_str: str) -> int:
    """
    Converts a file or data size given with a unit prefix into number of bytes.

    Example: "5 MB" --> 5000000

    Currently supported are "B", "KB", "MB", "GB", and "TB".

    Note:
        - The decimal prefix is used, not the binary one, i.e 1 KB = 1000 B and \
        not 1 KB = 1024 B.
        - Capitalization does not matter, i.e. "1 kb" is equal to "1 KB".

    Args:
        size_str (str): The size with unit prefix to convert.

    Raises:
        ValueError: If the size cannot be parsed. Mostly due to the unit not
            being one of the supoorted ones.

    Returns:
        int: The size as number of bytes.
    """
    size_str = size_str.strip().upper()
    size_units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}

    pattern = r"^(\d+(?:\.\d+)?)\s*(" + "|".join(size_units.keys()) + ")$"
    match = re.search(pattern, size_str)

    if match:
        value, unit = float(match.group(1)), match.group(2)
        return int(value * size_units[unit])

    raise ValueError(f"Invalid size format: '{size_str}'")


def calculate_required_number_of_chunks(
    max_chunk_size_in_memory: str,
    data_array: List[xr.DataArray],
) -> int:
    """
    This function calculates the number of chunks needed to process a
    collection of `xr.DataArray` not using more than `max_chunk_size_in_memory`
    bytes of memory.

    Args:
        max_chunk_size_in_memory (str): The desired chunk site with unit
            prefix, e.g. "100 MB"
        data_array (List[xr.DataArray]): The data arrays to be processed.

    Returns:
        int: The number of chunks used to process the array(s).
    """
    max_chunk_size_bytes = parse_size(max_chunk_size_in_memory)
    data_arrays_size = sum([x.nbytes for x in data_array])
    n_chunks = int(np.ceil(data_arrays_size / max_chunk_size_bytes))
    return n_chunks


def calculate_chunk_size_from_max_chunk_size_in_memory(
    max_chunk_memory_size: str, data_array: Union[xr.DataArray, List[xr.DataArray]]
) -> int:
    """
    Given that a block of memory can not be larger than `max_chunk_memory_size` bytes
    this function calculates how many rows of a xarray can be processed at once.

    Args:
        max_chunk_memory_size (str): The desired chunk site with unit
            prefix, e.g. "100 MB"
        data_array (Union[xr.DataArray, List[xr.DataArray]]): The data arrays
            to be processed.

    Returns:
        int: The number of rows that can be processed.
    """
    if not isinstance(data_array, list):
        data_array = [data_array]
    n_chunks = calculate_required_number_of_chunks(max_chunk_memory_size, data_array)
    chunk_size = max(int(data_array[0].shape[0] / n_chunks), 1)
    return chunk_size


def read_CSV_to_ndarray(file: str) -> NDArray[np.float64]:
    """
    Reads a cvs file into a xarray.

    Note:
        - The field delimiter is set to ','
        - A quote is surrounded by '|'
        - The first row can be a comment starting with '#'

    Args:
        file (str): The name of the csv file.

    Returns:
        NDArray[np.float64]: The xarray containing the data from the file.
    """
    import csv

    sources = []
    with open(file, newline="") as sourcefile:
        spam_reader = csv.reader(sourcefile, delimiter=",", quotechar="|")
        for row in spam_reader:
            if len(row) == 0:
                continue
            if row[0].startswith("#"):
                continue
            else:
                n_row = []
                for cell in row:
                    try:
                        value = float(cell)
                        n_row.append(value)
                    except ValueError:
                        pass
                sources.append(n_row)
    return np.array(sources, dtype=float)


def full_setter(self: object, state: Dict[str, Any]) -> None:
    self.__dict__ = state


def full_getter(self: object) -> Dict[str, Any]:
    state = self.__dict__
    return state


def get_spectral_sky_data(
    ra: NDArray[np.float_],
    dec: NDArray[np.float_],
    freq0: NDArray[np.float_],
    nfreq: int,
) -> NDArray[np.float_]:
    """
    Used to add the properties flux and frequency to a sample of
    point sources. The data is then returned as a structure that can be
    used to define a SkyModel.

    See :obj:`karabo.simulation.sky_model.SkyModel.add_point_sources`

    Args:
        ra (NDArray[np.float_]): A list of RA (right ascension) of the sources.
        dec (NDArray[np.float_]): Their declinations
        freq0 (NDArray[np.float_]): The main frequency of each source
        nfreq (int): Number of frequencies to sample. Must have the same length
            as the other arrays.

    Returns:
        NDArray[np.float_]: A sky model (n x 12 array) which has the following \
        entries set for each source:
            - [0]: right ascension
            - [1]: declination
            - [2]: stokes I flux
            - [6]: reference frequency
            - [7]: spectral index set to -200
    """
    dfreq_arr = np.linspace(-0.1, 0.1, 100)
    y_voigt = cast(NDArray[NPIntFloat], Voigt(dfreq_arr, 0, 0, 1, 0.01, 0.01))
    # y_gauss = Gauss(dfreq_arr, 0, 0, 1, 0.01)
    dfreq_sample = dfreq_arr[::nfreq]
    flux_sample = y_voigt[::nfreq]

    # Why do we add a frequency?
    freq_sample = freq0 + dfreq_sample * freq0
    sky_data = np.zeros((nfreq, 12))
    sky_data[:, 0] = ra
    sky_data[:, 1] = dec
    sky_data[:, 2] = flux_sample
    sky_data[:, 6] = freq_sample
    sky_data[:, 7] = -200
    return sky_data


def resample_spectral_lines(
    npoints: int,
    dfreq: NDArray[np.float_],
    spec_line: NDArray[np.float_],
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Downsamples a spectral line. The new line will have npoints.

    Args:
        npoints (int): Desired number of points after resampling. Must be samller
            than in the original sample.
        dfreq (NDArray[np.float_]): The frequencies samples fo the original line.
        spec_line (NDArray[np.float_]): The intensities of the original line.

    Returns:
        Tuple[NDArray[np.float_], NDArray[np.float_]]: A tuple of arrays where \
        [new_frequency_samples[], new_intensitiy_samples[]].
    """
    m = int(len(dfreq) / npoints)
    dfreq_sampled = dfreq[::m]
    line_sampled = spec_line[::m]
    return dfreq_sampled, line_sampled


def input_wrapper(
    msg: str,
    ret: str = "y",
) -> str:
    """
    Wrapper of standard `input` to define what return `ret` it will get during
    Unit-tests, since the test just stops otherwise.
    The environment variable 'SKIP_INPUT' or 'UNIT_TEST' must be set
    with an arbitrary value to return `ret`.

    Args:
        msg: input message
        ret: return value if 'SKIP_INPUT' or 'UNIT_TEST' is set, default='y'
    """
    if (
        os.environ.get("SKIP_INPUT") is not None
        or os.environ.get("UNIT_TEST") is not None
    ):
        return ret
    else:
        return input(msg)
