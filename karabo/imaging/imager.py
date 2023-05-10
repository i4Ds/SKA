from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from astropy.wcs import WCS
from distributed import Client
from numpy.typing import NDArray
from rascil.processing_components import create_visibility_from_ms
from rascil.workflows import (
    continuum_imaging_skymodel_list_rsexecute_workflow,
    create_visibility_from_ms_rsexecute,
)
from rascil.workflows.rsexecute.execution_support import rsexecute
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_func_python.image import image_gather_channels
from ska_sdp_func_python.imaging import (
    create_image_from_visibility,
    invert_visibility,
    remove_sumwt,
)
from ska_sdp_func_python.visibility import convert_visibility_to_stokesI

from karabo.error import KaraboError
from karabo.imaging.image import Image
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.visibility import Visibility
from karabo.util.dask import DaskHandler
from karabo.util.file_handle import FileHandle


class Imager:
    """Imager class provides imaging functionality using the visibilities
    of an observation with the help of RASCIL.

    In addition, it provides the calculation of the pixel coordinates of point sources.
    """

    def __init__(
        self,
        visibility: Visibility,
        imaging_npixel: Optional[int] = None,
        imaging_cellsize: Optional[float] = None,
        override_cellsize: bool = False,
    ) -> None:
        """Imager constructor.

        Args:
            visibility: Visibility object containing the visibilities of an observation.
            imaging_npixel: Number of pixels in ra, dec:
                Should be a composite of 2, 3, 5.
            imaging_cellsize: Cellsize (radians). Default is to calculate.
            override_cellsize: Override the cellsize if it is above
                the critical cellsize.
        """
        self.visibility = visibility
        self.imaging_npixel = imaging_npixel
        self.imaging_cellsize = imaging_cellsize
        self.override_cellsize = override_cellsize

    def get_dirty_image(
        self,
    ) -> Image:
        """Creates dirty image of `self.visibility` passed to `Imager`.

        Returns:
            Image: Dirty image of `self.visibility`.
        """
        block_visibilities = create_visibility_from_ms(self.visibility.ms_file.path)

        if len(block_visibilities) != 1:
            raise EnvironmentError("Visibilities are too large")
        ska_sdp_visibility = block_visibilities[0]
        file_handle = FileHandle(file_name="dirty", suffix=".fits")
        model = create_image_from_visibility(
            ska_sdp_visibility,
            npixel=self.imaging_npixel,
            cellsize=self.imaging_cellsize,
            override_cellsize=self.override_cellsize,
        )
        dirty, _ = invert_visibility(ska_sdp_visibility, model, context="2d")
        dirty.image_acc.export_to_fits(fits_file=f"{file_handle.path}")

        image = Image(path=file_handle)
        return image

    def imaging_rascil(
        self,
        client: Optional[Client] = None,
        use_dask: bool = False,
        n_threads: int = 1,
        use_cuda: bool = False,
        ingest_dd: List[int] = [0],
        ingest_vis_nchan: Optional[int] = None,
        ingest_chan_per_vis: int = 1,
        imaging_nchan: int = 1,
        imaging_w_stacking: bool = True,
        imaging_flat_sky: bool = False,
        imaging_uvmax: Optional[float] = None,
        imaging_uvmin: Optional[float] = None,
        imaging_dft_kernel: Optional[Literal["cpu_looped", "gpu_raw"]] = None,
        imaging_context: Literal["ng", "2d", "wg"] = "ng",
        clean_algorithm: Literal["hogbom", "msclean", "mmclean"] = "hogbom",
        clean_beam: Optional[Dict[Literal["bmaj", "bmin", "bpa"], float]] = None,
        clean_scales: List[int] = [0],
        clean_nmoment: int = 4,
        clean_nmajor: int = 5,
        clean_niter: int = 1000,
        clean_psf_support: int = 256,
        clean_gain: float = 0.1,
        clean_threshold: float = 1e-4,
        clean_component_threshold: Optional[float] = None,
        clean_component_method: Literal["fit", "extract"] = "fit",
        clean_fractional_threshold: float = 0.3,
        clean_facets: int = 1,
        clean_overlap: int = 32,
        clean_taper: Optional[Literal["linear", "tukey"]] = "tukey",
        clean_restore_facets: int = 1,
        clean_restore_overlap: int = 32,
        clean_restore_taper: Optional[Literal["linear", "tukey"]] = "tukey",
        clean_restored_output: Literal["list", "taylor", "integrated"] = "list",
    ) -> Tuple[Image, Image, Image]:
        """Starts imaging process using RASCIL.

        The process runs CLEAN on `self.visibility`.

        Args:
            client: Dask client
            use_dask: Use dask?
            n_threads: Number of dask threads
            use_cuda: Use cuda for nifty gridder?
            ingest_dd: Data descriptors in MS to read
                (all must have the same number of channels)
            ingest_vis_nchan: Number of channels in a single data descriptor in the MS
            ingest_chan_per_vis: Number of channels per vis (before any average)
            imaging_nchan: Number of channels per image
            imaging_w_stacking: Use the improved w stacking method in Nifty Gridder?
            imaging_flat_sky: If using a primary beam, normalise to flat sky?
            imaging_uvmax: Maximum uv (wavelengths)
            imaging_uvmin: Minimum uv (wavelengths)
            imaging_dft_kernel: DFT kernel: cpu_looped | gpu_raw
            imaging_context: Imaging context i.e. the gridder used 2d | ng
            clean_algorithm: Type of deconvolution algorithm
                (hogbom or msclean or mmclean)
            clean_beam: Clean beam: major axis, minor axis, position angle (deg)
            clean_scales: Scales for multiscale clean (pixels) e.g. [0, 6, 10]
            clean_nmoment: Number of frequency moments in mmclean
                (1 is a constant, 2 is linear, etc.)
            clean_nmajor: Number of major cycles in cip or ical
            clean_niter: Number of minor cycles in CLEAN (i.e. clean iterations)
            clean_psf_support: Half-width of psf used in cleaning (pixels)
            clean_component_method: Method to convert sources in image to skycomponents:
                'fit' in frequency or 'extract' actual values
            clean_fractional_threshold: Fractional stopping threshold for major cycle
            clean_facets: Number of overlapping facets in faceted clean
                (along each axis)
            clean_overlap: Overlap of facets in clean (pixels)
            clean_restore_facets: Number of overlapping facets in restore step
                (along each axis)
            clean_restore_overlap: Overlap of facets in restore step (pixels)
            clean_restored_output: Type of restored image output: taylor, list,
                or integrated

        Returns:
            Deconvolved image, Restored image, Residual image
        """
        if client and not use_dask:
            raise EnvironmentError("Client passed but use_dask is False")
        if use_dask:
            client = DaskHandler.get_dask_client()
        if client:
            print(client.cluster)
        # Set CUDA parameters
        if use_cuda:
            imaging_context = "wg"
        rsexecute.set_client(use_dask=use_dask, client=client, use_dlg=False)

        if ingest_vis_nchan is None:
            raise KaraboError("`ingest_vis_nchan` is None but must be of type 'int'.")

        blockviss = create_visibility_from_ms_rsexecute(
            msname=self.visibility.ms_file.path,
            nchan_per_vis=ingest_chan_per_vis,
            nout=ingest_vis_nchan // ingest_chan_per_vis,  # pyright: ignore
            dds=ingest_dd,
            average_channels=True,
        )

        blockviss = [
            rsexecute.execute(convert_visibility_to_stokesI)(bv) for bv in blockviss
        ]

        models = [
            rsexecute.execute(create_image_from_visibility)(
                bvis,
                npixel=self.imaging_npixel,
                nchan=imaging_nchan,
                cellsize=self.imaging_cellsize,
                override_cellsize=self.override_cellsize,
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            for bvis in blockviss
        ]
        # WAGG support for rascil does currently not work:
        # https://github.com/i4Ds/Karabo-Pipeline/issues/360
        if imaging_context == "wg":
            raise NotImplementedError("WAGG support for rascil does currently not work")

        result = continuum_imaging_skymodel_list_rsexecute_workflow(
            vis_list=blockviss,
            model_imagelist=models,
            context=imaging_context,
            threads=n_threads,
            wstacking=imaging_w_stacking,
            niter=clean_niter,
            nmajor=clean_nmajor,
            algorithm=clean_algorithm,
            gain=clean_gain,
            scales=clean_scales,
            fractional_threshold=clean_fractional_threshold,
            threshold=clean_threshold,
            nmoment=clean_nmoment,
            psf_support=clean_psf_support,
            restored_output=clean_restored_output,
            deconvolve_facets=clean_facets,
            deconvolve_overlap=clean_overlap,
            deconvolve_taper=clean_taper,
            restore_facets=clean_restore_facets,
            restore_overlap=clean_restore_overlap,
            restore_taper=clean_restore_taper,
            dft_compute_kernel=imaging_dft_kernel,
            component_threshold=clean_component_threshold,
            component_method=clean_component_method,
            flat_sky=imaging_flat_sky,
            clean_beam=clean_beam,
            clean_algorithm=clean_algorithm,
            imaging_uvmax=imaging_uvmax,
            imaging_uvmin=imaging_uvmin,
        )

        result = rsexecute.compute(result, sync=True)

        residual, restored, skymodel = result

        deconvolved = [sm.image for sm in skymodel]
        deconvolved_image_rascil = image_gather_channels(deconvolved)
        file_handle_deconvolved = FileHandle(file_name="deconvolved", suffix=".fits")
        deconvolved_image_rascil.image_acc.export_to_fits(
            fits_file=file_handle_deconvolved.path
        )
        deconvolved_image = Image(path=file_handle_deconvolved.path)

        if isinstance(restored, list):
            restored = image_gather_channels(restored)
        file_handle_restored = FileHandle(file_name="restored", suffix=".fits")
        restored.image_acc.export_to_fits(fits_file=file_handle_restored.path)
        restored_image = Image(path=file_handle_restored.path)

        residual = remove_sumwt(residual)
        if isinstance(residual, list):
            residual = image_gather_channels(residual)
        file_handle_residual = FileHandle(file_name="residual", suffix=".fits")
        residual.image_acc.export_to_fits(fits_file=file_handle_residual.path)
        residual_image = Image(path=file_handle_residual.path)

        return deconvolved_image, restored_image, residual_image

    @staticmethod
    def project_sky_to_image(
        sky: SkyModel,
        phase_center: Union[List[int], List[float]],
        imaging_cellsize: float,
        imaging_npixel: int,
        filter_outlier: bool = True,
        invert_ra: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Calculates the pixel coordinates `sky` sources as floats.

        If you want to have integer indices, just round them.

        Args:
            sky: `SkyModel` with the sources
            phase_center: [RA,DEC]
            imaging_cellsize: Image cellsize in radian (pixel coverage)
            imaging_npixel: Number of pixels of the image
            filter_outlier: Exclude source outside of image?
            invert_ra: Invert RA axis?

        Returns:
            Image-coordinates, `SkyModel` sources indices
        """

        # calc WCS args
        def radian_degree(rad: float) -> float:
            return rad * (180 / np.pi)

        cdelt = radian_degree(imaging_cellsize)
        crpix = np.floor((imaging_npixel / 2)) + 1

        # setup WCS
        w = WCS(naxis=2)
        w.wcs.crpix = np.array([crpix, crpix])  # coordinate reference pixel per axis
        ra_sign = -1 if invert_ra else 1
        w.wcs.cdelt = np.array(
            [ra_sign * cdelt, cdelt]
        )  # coordinate increments on sphere per axis
        w.wcs.crval = phase_center
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type

        # convert coordinates
        px, py = w.wcs_world2pix(sky[:, 0], sky[:, 1], 1)

        # check length to cover single source pre-filtering
        if len(px.shape) == 0 and len(py.shape) == 0:
            px, py = [px], [py]
            idxs = np.arange(sky.num_sources)
        # post processing, pre filtering before calling wcs.wcs_world2pix would be
        # more efficient, however this has to be done in the ra-dec space.
        # Maybe for future work!?
        elif filter_outlier:
            px_idxs = np.where(np.logical_and(px <= imaging_npixel, px >= 0))[0]
            py_idxs = np.where(np.logical_and(py <= imaging_npixel, py >= 0))[0]
            idxs = np.intersect1d(px_idxs, py_idxs)
            px, py = px[idxs], py[idxs]
        else:
            idxs = np.arange(sky.num_sources)
        img_coords = np.array([px, py])

        return img_coords, idxs
