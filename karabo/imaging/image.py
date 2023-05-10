from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray
from rascil.apps.imaging_qa.imaging_qa_diagnostics import power_spectrum
from scipy.interpolate import RegularGridInterpolator

from karabo.karabo_resource import KaraboResource
from karabo.util.file_handle import FileHandle, check_ending
from karabo.util.plotting_util import get_slices

# store and restore the previously set matplotlib backend,
# because rascil sets it to Agg (non-GUI)
previous_backend = matplotlib.get_backend()

matplotlib.use(previous_backend)


class Image(KaraboResource):
    """Image proxy object providing some utility features."""

    def __init__(
        self,
        path: Union[str, FileHandle],
        **kwargs: Any,
    ) -> None:
        """Image constructor.

        Args:
            path: Path to .fits image.
        """
        if isinstance(path, FileHandle):
            _path = path.path
        else:
            _path = path
        self.path = _path
        self.__name = self.path.split(os.path.sep)[-1]
        self.data: NDArray[np.float_]
        self.header: fits.header.Header
        self.data, self.header = fits.getdata(self.path, ext=0, header=True, **kwargs)

    @staticmethod
    def read_from_file(path: str) -> Image:
        """Static object instantiation.

        Args:
            path: Path to .fits image.

        Returns:
            Image proxy.
        """
        return Image(path=path)

    @property
    def data(self) -> NDArray[np.float_]:
        """Fits data getter-property.

        Returns:
            Data of image proxy.
        """
        return self._data

    @data.setter
    def data(self, new_data: NDArray[np.float_]) -> None:
        """Fits data setter-property.

        Args:
            new_data: Data for update.
        """
        self._data = new_data
        if hasattr(self, "header"):
            self._update_header_after_resize()

    def write_to_file(
        self,
        path: str,
        overwrite: bool = False,
    ) -> None:
        """Write `Image` to `path`  as .fits file.

        Args:
            path: Path to write the file.
            overwrite: Overwrite .fits file if already exist?
        """
        check_ending(path=path, ending=".fits")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        fits.writeto(
            filename=path,
            data=self.data,
            header=self.header,
            overwrite=overwrite,
        )

    def header_has_parameters(
        self,
        parameters: List[str],
    ) -> bool:
        """Checks if header has the parameters `parameters`.

        Args:
            parameters: Parameters to check.

        Returns:
            True if header has ALL parameters, else False
        """
        for parameter in parameters:
            if parameter not in self.header:
                return False
        return True

    def get_squeezed_data(self) -> NDArray[np.float_]:
        """Squeeze `Image` data.

        Returns:
            Squeezed data
        """
        return np.squeeze(self.data[:1, :1, :, :])

    def resample(
        self,
        shape: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        """Resamples the `Image.data`.

        Resamples the image to the given shape using SciPy's RegularGridInterpolator
        for bilinear interpolation. See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        Args:
            shape: Desired image shape
            kwargs: kwargs for interpolation function
        """
        new_data = np.empty(
            (self.data.shape[0], 1, shape[0], shape[1]), dtype=self.data.dtype
        )

        for c in range(self.data.shape[0]):
            y = np.arange(self.data.shape[2])
            x = np.arange(self.data.shape[3])
            interpolator = RegularGridInterpolator(
                (y, x),
                self.data[c, 0],
                **kwargs,
            )

            new_x = np.linspace(0, self.data.shape[3] - 1, shape[1])
            new_y = np.linspace(0, self.data.shape[2] - 1, shape[0])
            new_points = np.array(np.meshgrid(new_y, new_x)).T.reshape(-1, 2)
            new_data[c] = interpolator(new_points).reshape(shape[0], shape[1])

        self.data = new_data

    def _update_header_after_resize(self) -> None:
        """Updates the header shape."""
        old_shape = (self.header["NAXIS2"], self.header["NAXIS1"])
        new_shape = (self.data.shape[2], self.data.shape[3])
        self.header["NAXIS1"] = new_shape[1]
        self.header["NAXIS2"] = new_shape[0]

        self.header["CRPIX1"] = (new_shape[1] + 1) / 2
        self.header["CRPIX2"] = (new_shape[0] + 1) / 2

        self.header["CDELT1"] = self.header["CDELT1"] * old_shape[1] / new_shape[1]
        self.header["CDELT2"] = self.header["CDELT2"] * old_shape[0] / new_shape[0]

    def plot(
        self,
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        colobar_label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: Optional[str] = "jet",
        origin: Optional[Literal["lower", "upper"]] = "lower",
        wcs_enabled: bool = True,
        invert_xaxis: bool = False,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Plots the `Image.data`.

        Args:
            title: Title of the colormap
            xlim: RA-limit x-axis
            ylim: DEC-limit y-axis
            figsize: Figsize
            colobar_label: Colorbar label
            xlabel: xlabel
            ylabel: ylabel
            cmap: matplotlib color-map
            origin: Place the [0, 0] index of the array in
                the upper left or lower left corner of the Axes
            wcs_enabled: Use wcs transformation?
            invert_xaxis: Invert the xaxis?
            filename: Set to path/fname to save figure
                (set extension to fname to overwrite .png default)
            kwargs: matplotlib kwargs for scatter & Collections,
                e.g. customize `s`, `vmin` or `vmax`
        """

        if wcs_enabled:
            wcs = WCS(self.header)

            slices = get_slices(wcs=wcs)

            # create dummy xlim or ylim if only one is set for conversion
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

        if wcs_enabled:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection=wcs, slices=slices)
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(self.data[0][0], cmap=cmap, origin=origin, **kwargs)
        ax.grid()
        fig.colorbar(im, label=colobar_label)

        if title is not None:
            ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if invert_xaxis:
            ax.invert_xaxis()
        if filename is not None:
            fig.savefig(filename)
        plt.show(block=False)
        plt.pause(1)

    def get_dimensions_of_image(self) -> List[int]:
        """Get the sizes of the dimensions of `Image`.

        Returns:
            Dimensions of `Image`
        """
        result = []
        dimensions = self.header["NAXIS"]
        for dim in np.arange(0, dimensions, 1):
            result.append(self.header[f"NAXIS{dim + 1}"])
        return result

    def get_phase_center(self) -> Tuple[float, float]:
        """Gets the physical value of the reference pixel.

        Returns:
            RA, DEC
        """
        return float(self.header["CRVAL1"]), float(self.header["CRVAL2"])

    def has_beam_parameters(self) -> bool:
        """Check if header has BMAJ, BMIN and BPA parameters.

        Returns:
            True if header has them, else False
        """
        return self.header_has_parameters(
            ["BMAJ", "BMIN", "BPA"],
        )

    def get_quality_metric(self) -> Dict[str, Any]:
        """Provides some image statistics.

        - Shape of Image --> 'shape'
        - Max Value --> 'max'
        - Min Value --> 'min'
        - Max Value absolute --> 'max-abs'
        - Root mean square (RMS) --> 'rms'
        - Sum of values --> 'sum'
        - Median absolute --> 'median-abs'
        - Median absolute deviation median --> 'median-abs-dev-median'
        - Median --> 'median'
        - Mean --> 'mean'

        Returns:
            Dict of all mentioned statistics.
        """
        # same implementation as RASCIL
        image_stats = {
            "shape": str(self.data.shape),
            "max": np.max(self.data),
            "min": np.min(self.data),
            "max-abs": np.max(np.abs(self.data)),
            "rms": np.std(self.data),
            "sum": np.sum(self.data),
            "median-abs": np.median(np.abs(self.data)),
            "median-abs-dev-median": np.median(
                np.abs(self.data - np.median(self.data))
            ),
            "median": np.median(self.data),
            "mean": np.mean(self.data),
        }

        return image_stats

    def get_power_spectrum(
        self,
        resolution: float = 5.0e-4,
        signal_channel: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate the power spectrum of this image.

        Args:
            resolution: Resolution in radians needed for conversion from Jy to Kelvin
            signal_channel: channel containing both signal and noise

        Returns:
            profile: Brightness temperature for each angular scale in Kelvin,
            theta_axis: Angular scale data in degrees
        """
        profile, theta = power_spectrum(self.path, resolution, signal_channel)
        return profile, theta

    def plot_power_spectrum(
        self,
        resolution: float = 5.0e-4,
        signal_channel: Optional[int] = None,
        save_png: bool = False,
    ) -> None:
        """Plot the power spectrum of this image.

        Args:
            resolution: Resolution in radians needed for conversion from Jy to Kelvin
            signal_channel: Channel containing both signal and noise
            save_png: Should result be saved as .png? (in `self.path`)
        """
        profile, theta = self.get_power_spectrum(resolution, signal_channel)
        plt.clf()

        plt.plot(theta, profile)
        plt.gca().set_title(
            f"Power spectrum of {self.__name if self.__name is not None else ''} image"
        )
        plt.gca().set_xlabel("Angular scale [degrees]")
        plt.gca().set_ylabel("Brightness temperature [K]")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().set_ylim(1e-6 * np.max(profile), 2.0 * np.max(profile))
        plt.tight_layout()

        if save_png:
            power_spectrum_name = (
                self.__name if self.__name is not None else uuid.uuid4()
            )
            plt.savefig(f"./power_spectrum_{power_spectrum_name}")
        plt.show(block=False)
        plt.pause(1)

    def get_cellsize(self) -> np.float64:
        """Get cellsize of `Image`.

        Assumes square-pixels.

        Returns:
            Cellsize
        """
        cdelt1 = self.header["CDELT1"]
        cdelt2 = self.header["CDELT2"]
        if not isinstance(cdelt1, float) or not isinstance(cdelt2, float):
            raise ValueError(
                "CDELT1 & CDELT2 in header are expected to be of type float."
            )
        if np.abs(cdelt1) != np.abs(cdelt2):
            logging.warning(
                "Non-square pixels are not supported, continue with `cdelt1`."
            )
        cellsize = cast(np.float64, np.deg2rad(np.abs(cdelt1)))
        return cellsize

    def get_wcs(self) -> WCS:
        """Creates `astropy.wcs.WCS` from `Image` header.

        Returns:
            WCS instance
        """
        return WCS(self.header)

    def get_2d_wcs(
        self,
        invert_ra: bool = True,
    ) -> WCS:
        """Creates a 2-dimensional `astropy.wcs.WCS` instance of `Image`.

        Args:
            invert_ra: Invert RA?

        Returns:
            WCS instance
        """
        wcs = WCS(naxis=2)

        def radian_degree(rad: np.float64) -> np.float64:
            return rad * (180 / np.pi)

        cdelt = radian_degree(self.get_cellsize())
        crpix = np.floor((self.get_dimensions_of_image()[0] / 2)) + 1
        wcs.wcs.crpix = np.array([crpix, crpix])
        ra_sign = -1 if invert_ra else 1
        wcs.wcs.cdelt = np.array([ra_sign * cdelt, cdelt])
        wcs.wcs.crval = [self.header["CRVAL1"], self.header["CRVAL2"]]
        wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type
        return wcs
