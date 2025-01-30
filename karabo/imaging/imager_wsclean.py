from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Union

from typing_extensions import override

from karabo.imaging.image import Image
from karabo.imaging.imager_base import (
    DirtyImager,
    DirtyImagerConfig,
    ImageCleaner,
    ImageCleanerConfig,
)
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler

_WSCLEAN_BINARY = "wsclean"


def _get_command_prefix(tmp_dir: str) -> str:
    return (
        # wsclean always uses the current directory as the working directory
        f"cd {tmp_dir} && "
        # Avoids the following wsclean error:
        # This software was linked to a multi-threaded version of OpenBLAS.
        # OpenBLAS multi-threading interferes with other multi-threaded parts of
        # the code, which has a severe impact on performance. Please disable
        # OpenBLAS multi-threading by setting the environment variable
        # OPENBLAS_NUM_THREADS to 1.
        "OPENBLAS_NUM_THREADS=1 "
    )


class WscleanDirtyImager(DirtyImager):
    """Dirty imager based on the WSClean library.

    WSClean is integrated by calling the wsclean command line tool.
    The parameters in the config (DirtyImagerConfig) attribute are passed to wsclean.
    Use the create_image_custom_command function if you need to set params
    not available in DirtyImagerConfig.

    Attributes:
        config (DirtyImagerConfig): Config containing parameters for
            dirty imaging

    """

    TMP_PREFIX_DIRTY = "WSClean-dirty-"
    TMP_PURPOSE_DIRTY = "Disk cache for WSClean dirty images"

    OUTPUT_FITS_DIRTY = "wsclean-dirty.fits"

    def __init__(self, config: DirtyImagerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (DirtyImagerConfig): see config attribute

        """
        super().__init__()
        self.config = config

    @override
    def create_dirty_image(
        self,
        visibility: Visibility,
        /,
        *,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if visibility.format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility.format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )
        # TODO combine_across_frequencies
        # -channels-out <count>?
        if self.config.combine_across_frequencies is False:
            raise NotImplementedError(
                "combine_across_frequencies=False is currently not supported "
                "for the WSClean imager."
            )

        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_DIRTY,
            purpose=self.TMP_PURPOSE_DIRTY,
        )
        command = _get_command_prefix(tmp_dir) + (
            f"{_WSCLEAN_BINARY} "
            f"-size {self.config.imaging_npixel} {self.config.imaging_npixel} "
            f"-scale {math.degrees(self.config.imaging_cellsize)}deg "
            f"{visibility.path}"
        )
        print(f"WSClean command: [{command}]")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        default_output_fits_path = os.path.join(tmp_dir, self.OUTPUT_FITS_DIRTY)
        if output_fits_path is None:
            output_fits_path = default_output_fits_path
        else:
            shutil.copyfile(default_output_fits_path, output_fits_path)

        return Image(path=output_fits_path)


# TODO Set kw_only=True after update to Python 3.10
# Right now, if one inherited superclass has a default-argument, you have to set
# defaults for all your attributes as well.
@dataclass
class WscleanImageCleanerConfig(ImageCleanerConfig):
    """Config / parameters of a WscleanImageCleaner.

    Adds parameters specific to WscleanImageCleaner.

    Attributes:
        niter (Optional[int]): Maximum number of clean iterations to perform.
            Defaults to 50000.
        mgain (Optional[float]): Cleaning gain for major iterations: Ratio of peak that
            will be subtracted in each major iteration. To use major iterations, 0.85 is
            a good value. Defaults to 0.8.
        auto_threshold (Optional[int]): Relative clean threshold. Estimate noise level
            using a robust estimator and stop at sigma x stddev. Defaults to 3.

    """

    niter: Optional[int] = 50000
    mgain: Optional[float] = 0.8
    auto_threshold: Optional[int] = 3


class WscleanImageCleaner(ImageCleaner):
    """Image cleaner based on the WSClean library.

    WSClean is integrated by calling the wsclean command line tool.
    The parameters in the config (WscleanImageCleanerConfig) attribute
    are passed to wsclean.
    Use the create_image_custom_command function if you need to set params
    not available in WscleanImageCleanerConfig.
    Parameters in the config that are explicitly set to None will not be passed to the
    command line tool, which will then resort to its own default values.

    Attributes:
        config (WscleanImageCleanerConfig): Config containing parameters for
            WSClean image cleaning.

    """

    TMP_PREFIX_CLEANED = "WSClean-cleaned-"
    TMP_PURPOSE_CLEANED = "Disk cache for WSClean cleaned images"

    OUTPUT_FITS_CLEANED = "wsclean-image.fits"

    def __init__(self, config: WscleanImageCleanerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (WscleanImageCleanerConfig): see config attribute

        """
        super().__init__()
        self.config = config

    @override
    def create_cleaned_image(
        self,
        visibility: Visibility,
        /,
        *,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if visibility.format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility.format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )

        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_CLEANED,
            purpose=self.TMP_PURPOSE_CLEANED,
        )
        prefix = "pre_existing"
        if dirty_fits_path is not None:
            shutil.copyfile(
                dirty_fits_path,
                os.path.join(tmp_dir, f"{prefix}-dirty.fits"),
            )
        command = _get_command_prefix(tmp_dir) + (
            f"{_WSCLEAN_BINARY} "
            + (f"-reuse-dirty {prefix} " if dirty_fits_path is not None else "")
            + f"-size {self.config.imaging_npixel} {self.config.imaging_npixel} "
            + f"-scale {math.degrees(self.config.imaging_cellsize)}deg "
            + (f"-niter {self.config.niter} " if self.config.niter is not None else "")
            + (f"-mgain {self.config.mgain} " if self.config.mgain is not None else "")
            + (
                f"-auto-threshold {self.config.auto_threshold} "
                if self.config.auto_threshold is not None
                else ""
            )
            + str(visibility.path)
        )
        print(f"WSClean command: [{command}]")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        default_output_fits_path = os.path.join(tmp_dir, self.OUTPUT_FITS_CLEANED)
        if output_fits_path is None:
            output_fits_path = default_output_fits_path
        else:
            shutil.copyfile(default_output_fits_path, output_fits_path)

        return Image(path=output_fits_path)


TMP_PREFIX_CUSTOM = "WSClean-custom-"
TMP_PURPOSE_CUSTOM = "Disk cache for WSClean custom command images"


def create_image_custom_command(
    command: str,
    output_filenames: Union[str, List[str]] = "wsclean-image.fits",
) -> Union[Image, List[Image]]:
    """Create a dirty or cleaned image using your own command.

    Allows the use of the full WSClean functionality with all parameters.
    Command has to start with 'wsclean '.
    The working directory the command runs in will be a temporary directory.
    Use absolute paths to reference files or directories like the measurement set.

    Args:
        command: Command to execute.
            Example: wsclean -size 2048 2048
            -scale 0.00222222deg -niter 50000 -mgain 0.8
            -abs-threshold 100µJy /tmp/measurements.MS
        output_filenames: WSClean output filename(s)
            (relative to the working directory) that should be returned
            as Image objects. Can be a string for one file or a list of strings
            for multiple files.
            Example 1: "wsclean-image.fits"
            Example 2: ['wsclean-image.fits', 'wsclean-residual.fits']

    Returns:
        - If output_filenames is a **string**, returns an Image object of the file \
            output_filenames.
        - If output_filenames is a **list of strings**, returns a list of \
            Image objects, one object per filename in output_filenames.

    """

    tmp_dir = FileHandler().get_tmp_dir(
        prefix=TMP_PREFIX_CUSTOM,
        purpose=TMP_PURPOSE_CUSTOM,
    )
    expected_command_prefix = f"{_WSCLEAN_BINARY} "
    if not command.startswith(expected_command_prefix):
        raise ValueError(
            "Unexpected command. Expecting command to start with "
            f'"{expected_command_prefix}".'
        )
    command = _get_command_prefix(tmp_dir) + command
    print(f"WSClean command: [{command}]")
    completed_process = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        # Raises exception on return code != 0
        check=True,
    )
    print(f"WSClean output:\n[{completed_process.stdout}]")

    if isinstance(output_filenames, str):
        return Image(path=os.path.join(tmp_dir, output_filenames))
    else:
        return [
            Image(path=os.path.join(tmp_dir, output_filename))
            for output_filename in output_filenames
        ]
