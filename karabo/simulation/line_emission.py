import os
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, cast, overload

import astropy.units as u
import matplotlib
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.image.image_model import Image as RASCILImage
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from karabo.imaging.image import Image, ImageMosaicker
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.interferometer import FilterUnits, InterferometerSimulation
from karabo.simulation.line_emission_helpers import convert_frequency_to_z
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend

CircleSkyRegion = namedtuple("CircleSkyRegion", ["center", "radius"])


def generate_gaussian_beam_data(
    fwhm_pixels: float,
    x_size: int,
    y_size: int,
) -> NDArray[np.float_]:
    """Given a FWHM in pixel units, and a size in x and y coordinates,
    return a 2D array of shape (x_size, y_size) containing normalized Gaussian values
    (such that the central value of the 2D array is 1.0).
    """
    sigma = fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))
    gauss_kernel = Gaussian2DKernel(
        sigma,
        x_size=x_size,
        y_size=y_size,
    )
    beam = cast(NDArray[np.float_], gauss_kernel.array)
    beam = beam / np.max(beam)

    return beam


# Reference values below were obtained for a MeerKAT-like beam
REFERENCE_FWHM_DEGREES = 1.8
REFERENCE_FREQUENCY_HZ = 8e8


def gaussian_beam_fwhm_for_frequency(
    desired_frequency: float,
    reference_fwhm_degrees: float = REFERENCE_FWHM_DEGREES,
    reference_frequency_Hz: float = REFERENCE_FREQUENCY_HZ,
) -> float:
    return reference_fwhm_degrees * reference_frequency_Hz / desired_frequency


@overload
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: List[NDArray[np.float_]],
    should_perform_primary_beam_correction: Literal[True] = ...,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    ...


@overload
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = ...,
    should_perform_primary_beam_correction: Literal[False] = ...,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    ...


@overload
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = ...,
    should_perform_primary_beam_correction: bool = ...,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    ...


def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = None,
    should_perform_primary_beam_correction: bool = False,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    """Perform a line emission simulation, to compute visibilities and dirty images.
    A line emission simulation involves assuming every source in the input SkyModel
    only emits within one frequency channel.

    If requested, include primary beam effects into the visibilities and dirty images.
    And again, if desired, perform a primary beam correction on the final dirty images.

    For OSKAR, the provided primary beams will only be used for correction.
    For the actual primary beam effect in OSKAR,
    set the relevant parameter in the InterferometerSimulation constructor.

    For RASCIL, the provided primary beams are used
    for both the primary beam effect and its correction.
    """
    print(f"Selected backend: {simulator_backend}")

    output_base_directory = Path(output_base_directory)
    # If output filepath does not exist, mkdir
    print(f"Creating {output_base_directory} directory if it does not exist yet.")
    output_base_directory.mkdir(exist_ok=True, parents=True)

    # Compute frequency channels
    frequency_channel_starts = np.linspace(
        observation_details.start_frequency_hz,
        observation_details.start_frequency_hz
        + observation_details.frequency_increment_hz
        * observation_details.number_of_channels,
        num=observation_details.number_of_channels,
        endpoint=False,
    )

    # Verify that, if primary beams are provided,
    # we have one primary beam per frequency channel
    if primary_beams is not None:
        assert (
            len(primary_beams) == observation_details.number_of_channels
        ), f"""Did not provide same number of primary beams
            as number of desired frequency channels:
        {len(primary_beams)}, {observation_details.number_of_channels}"""

    # Create observation instance to be used for each pointing and each channel
    observation = deepcopy(observation_details)

    print("Computing visibilities...")

    # Loop through pointings
    visibilities: List[List[Visibility]] = []

    for index_freq, frequency_start in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        visibilities.append([])

        for index_p, p in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            center = p.center
            radius = p.radius.to(u.deg).value
            # Create observation details
            observation.phase_centre_ra_deg = center.ra.deg
            observation.phase_centre_dec_deg = center.dec.deg
            observation.number_of_channels = 1  # For line emission
            observation.start_frequency_hz = frequency_start

            # Convert provided primary beam to required image format
            if primary_beams is None:
                primary_beam = None
            else:
                # Currently supported backend for custom primary beams: RASCIL
                primary_beam = create_image(
                    npixel=dirty_imager_config.imaging_npixel,
                    cellsize=dirty_imager_config.imaging_cellsize,
                    phasecentre=center,
                    polarisation_frame=PolarisationFrame(
                        "stokesI"
                    ),  # TODO support full stokes
                    frequency=observation.start_frequency_hz,
                    channel_bandwidth=observation.frequency_increment_hz,
                    nchan=1,
                )

                # TODO support full stokes,
                # instead of hardcoding index 0 for the polarisation
                primary_beam["pixels"][0][0] = primary_beams[index_freq]

            # Filter sky based on pointing and on frequency channel
            filtered_sky = sky_model.filter_by_radius_euclidean_flat_approximation(
                inner_radius_deg=0,
                outer_radius_deg=radius,
                ra0_deg=center.ra.deg,
                dec0_deg=center.dec.deg,
            )
            z_min = convert_frequency_to_z(
                frequency_start + observation.frequency_increment_hz
            )
            z_max = convert_frequency_to_z(frequency_start)

            filtered_sky = filtered_sky.filter_by_column(
                col_idx=13,
                min_val=z_min,
                max_val=z_max,
            )

            assert (
                filtered_sky.num_sources > 0
            ), f"""For frequency channel {index_freq}
                    and pointing {index_p}, there are 0 sources in the sky model.
                    Setting visibility to None, and skipping analysis."""

            vis = interferometer.run_simulation(
                telescope=telescope,
                sky=filtered_sky,
                observation=observation,
                backend=simulator_backend,
                primary_beam=primary_beam,
                visibility_format="MS",
                visibility_path=os.path.join(
                    output_base_directory,
                    f"visibilities_f{index_freq}_p{index_p}.MS",
                ),
            )  # type: ignore[call-overload]

            visibilities[-1].append(vis)

    assert len(visibilities) == observation_details.number_of_channels
    assert len(visibilities[0]) == len(pointings)

    print("Creating dirty images from visibilities...")

    dirty_images: List[List[Image]] = []
    for index_freq, _ in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        dirty_images.append([])
        for index_p, _ in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            vis = visibilities[index_freq][index_p]

            if simulator_backend is SimulatorBackend.OSKAR:
                backend = "OSKAR"
            else:
                backend = "RASCIL"
            dirty_imager = RascilDirtyImager(
                RascilDirtyImagerConfig(
                    imaging_npixel=dirty_imager_config.imaging_npixel,
                    imaging_cellsize=dirty_imager_config.imaging_cellsize,
                    combine_across_frequencies=dirty_imager_config.combine_across_frequencies,  # noqa: E501
                )
            )
            dirty = dirty_imager.create_dirty_image(
                vis,
                output_fits_path=os.path.join(
                    output_base_directory, f"dirty_{backend}_{index_p}.fits"
                ),
            )

            # Perform beam correction here, if requested
            # NOTE we are correcting each pointing before returning the dirty images,
            # i.e. before creating any mosaics of pointings
            if should_perform_primary_beam_correction is True:
                # Verify that, if primary beam correction is requested,
                # the corresponding primary beams are provided
                assert (
                    primary_beams is not None
                ), "Primary beam correction was requested, but no beams provided."

                primary_beam = primary_beams[index_freq]
                dirty.data[0][0] /= primary_beam  # TODO handle full stokes

            dirty_images[-1].append(dirty)

            # dirty.data.shape meaning:
            # (frequency channels, polarisations, pixels_x, pixels_y)
            assert dirty.data.ndim == 4
            # I.e. only one frequency channel,
            # since we are performing a line emission analysis
            assert dirty.data.shape[0] == 1

    assert len(dirty_images) == observation_details.number_of_channels
    assert len(dirty_images[0]) == len(
        pointings
    ), f"{len(dirty_images[0])}, {len(pointings)}"

    return visibilities, dirty_images


if __name__ == "__main__":
    # This executes an example line emission pipeline
    # with a sample sky and example simulation parameters
    from karabo.data.external_data import HISourcesSmallCatalogDownloadObject
    from karabo.util.file_handler import FileHandler

    try:
        matplotlib.use("tkagg")
    except Exception as e:
        print(e)

    simulator_backend = SimulatorBackend.RASCIL

    if simulator_backend == SimulatorBackend.OSKAR:
        telescope = Telescope.constructor("SKA1MID", backend=simulator_backend)
    elif simulator_backend == SimulatorBackend.RASCIL:
        telescope = Telescope.constructor("MID", backend=simulator_backend)

    # Configuration parameters
    # Whether to include primary beam into vis and dirty images
    should_apply_primary_beam = True
    # Whether to correct for the primary beam in the dirty images before returning them
    should_perform_primary_beam_correction = True

    output_base_directory = Path(
        FileHandler().get_tmp_dir(
            prefix="line-emission-",
            purpose="Example line emission simulation",
        )
    )

    pointings = [
        CircleSkyRegion(
            radius=2 * u.deg, center=SkyCoord(ra=20, dec=-30, unit="deg", frame="icrs")
        ),
        CircleSkyRegion(
            radius=2 * u.deg,
            center=SkyCoord(ra=20, dec=-31.4, unit="deg", frame="icrs"),
        ),
        CircleSkyRegion(
            radius=2 * u.deg,
            center=SkyCoord(ra=21.4, dec=-30, unit="deg", frame="icrs"),
        ),
        CircleSkyRegion(
            radius=2 * u.deg,
            center=SkyCoord(ra=21.4, dec=-31.4, unit="deg", frame="icrs"),
        ),
    ]

    # The number of time steps is then determined as total_length / integration_time.
    observation_length = timedelta(seconds=10000)  # 14400 = 4hours
    integration_time = timedelta(seconds=1000)

    # Load catalog of sources
    catalog_path = HISourcesSmallCatalogDownloadObject().get()
    sky = SkyModel.get_sky_model_from_h5_to_xarray(
        path=catalog_path,
    )

    # Define observation channels and duration
    observation = Observation(
        start_date_and_time=datetime(2000, 3, 20, 12, 6, 39),
        length=observation_length,
        number_of_time_steps=int(
            observation_length.total_seconds() / integration_time.total_seconds()
        ),
        start_frequency_hz=7e8,
        frequency_increment_hz=8e7,
        number_of_channels=2,
    )

    # Imaging details
    npixels = 4096
    image_width_degrees = 2
    cellsize_radians = np.radians(image_width_degrees) / npixels
    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=npixels,
        imaging_cellsize=cellsize_radians,
    )

    # Create interferometer simulation
    beam_type: Literal["Gaussian beam", "Isotropic beam"]
    primary_beams: Optional[List[RASCILImage]] = None

    # Compute frequency channels
    frequency_channel_starts = np.linspace(
        observation.start_frequency_hz,
        observation.start_frequency_hz
        + observation.frequency_increment_hz * observation.number_of_channels,
        num=observation.number_of_channels,
        endpoint=False,
    )

    if should_apply_primary_beam:
        # Options accepted by OSKAR:
        # "Aperture array", "Isotropic beam", "Gaussian beam", "VLA (PBCOR)"
        # This parameter is not used by RASCIL
        beam_type = "Gaussian beam"
        gaussian_fwhm = REFERENCE_FWHM_DEGREES
        gaussian_ref_freq = REFERENCE_FREQUENCY_HZ

        primary_beams = []
        # RASCIL supports custom primary beams
        # Here we create a sample beam (Gaussian)
        # as a 2D np.array of shape (npixels, npixels)
        for frequency in frequency_channel_starts:
            fwhm_degrees = gaussian_beam_fwhm_for_frequency(frequency)
            fwhm_pixels = fwhm_degrees / np.degrees(
                dirty_imager_config.imaging_cellsize
            )

            primary_beam = generate_gaussian_beam_data(
                fwhm_pixels=fwhm_pixels,
                x_size=dirty_imager_config.imaging_npixel,
                y_size=dirty_imager_config.imaging_npixel,
            )
            primary_beams.append(primary_beam)
    else:
        beam_type = "Isotropic beam"
        gaussian_fwhm = 0
        gaussian_ref_freq = 0

    # Instantiate interferometer
    # Leave time_average_sec as 10, since OSKAR examples use 10.
    # Not sure of the meaning of this parameter.
    interferometer = InterferometerSimulation(
        time_average_sec=10,
        ignore_w_components=True,
        uv_filter_max=10000,
        uv_filter_units=FilterUnits.Metres,
        use_gpus=True,
        station_type=beam_type,
        gauss_beam_fwhm_deg=gaussian_fwhm,
        gauss_ref_freq_hz=gaussian_ref_freq,
        use_dask=False,
    )

    visibilities, dirty_images = line_emission_pipeline(
        output_base_directory=output_base_directory,
        pointings=pointings,
        sky_model=sky,
        observation_details=observation,
        telescope=telescope,
        interferometer=interferometer,
        simulator_backend=simulator_backend,
        dirty_imager_config=dirty_imager_config,
        primary_beams=primary_beams,
        should_perform_primary_beam_correction=should_perform_primary_beam_correction,
    )

    dirty_images[0][0].plot(
        block=True,
        vmin=0,
        vmax=2e-7,
        title="Dirty image for pointing 0 and channel 0",
    )

    # Overlay SkyModel onto dirty image
    dirty_images[0][0].plot_side_by_side_with_skymodel(
        sky=sky,
        block=True,
        vmin_sky=0,
        vmax_sky=2e-6,
        vmin_image=0,
        vmax_image=2e-7,
    )

    dirty_images[0][0].overplot_with_skymodel(
        sky=sky,
        block=True,
        vmin_image=0,
        vmax_image=2e-7,
    )

    # Create mosaics of pointings for each frequency channel
    print("Creating mosaic of images for each frequency channel")
    mosaicker = ImageMosaicker()

    mosaics = []
    for i in range(observation.number_of_channels):
        mosaic, footprint = mosaicker.mosaic(dirty_images[i])
        mosaics.append(mosaic)

        mosaic.plot(
            filename=str(output_base_directory / f"mosaic_{i}.png"),
            block=True,
            vmin=0,
            vmax=2e-7,
            title=f"Mosaic for channel {i}",
        )

    # Add all mosaics across frequency channels to create one final mosaic image
    summed_mosaic = Image(
        data=np.sum([m.data for m in mosaics]),
        header=mosaics[0].header,
    )
    summed_mosaic.plot(
        filename=str(output_base_directory / "summed_mosaic.png"),
        block=True,
        vmin=0,
        vmax=2e-7,
        title="Summed mosaic across channels",
    )

    # Overlay SkyModel onto dirty image
    summed_mosaic.overplot_with_skymodel(
        sky=sky,
        filename=str(output_base_directory / "summed_mosaic_sources_overlay.png"),
        block=True,
        vmin_image=0,
        vmax_image=2e-7,
    )
