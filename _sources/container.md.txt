# Containers

## Docker Images

We provide for the Karabo-pipeline [Docker images](https://www.docker.com/resources/what-container/#:~:text=A%20Docker%20container%20image%20is,tools%2C%20system%20libraries%20and%20settings.) which are hosted by the [ghcr.io](https://github.com/features/packages) registry. An overview of all available images is [here](https://github.com/i4ds/Karabo-Pipeline/pkgs/container/karabo-pipeline), if a specific version and not simply `latest` is desired. Starting from `karabo@v0.15.0`, all versions should be available. Provided you have docker, the image can be pulled as follows:

```shell
docker pull ghcr.io/i4ds/karabo-pipeline:latest
```

Docker images have the advantage that the packages needed for Karabo-pipeline are already pre-installed and you can usually run them on other operating systems. So in case the dependency resolving of older Karabo installations is not up to date anymore, with Docker images you don't have to worry as the installation process has already been performed. In addition, Docker images can easily transform into other containers like Singularity or Sarus, which are often used in HPC-clusters.

## Launch a Docker Container

What the possibilities using Docker are is far too extensive to describe here. We refer to the official [Docker reference](https://docs.docker.com/reference/) for this. We only show here a minimal example of how Docker could be used, so you can use e.g. a [Jupyter Notebook](https://jupyter.org/) with sample code and an existing Karabo environment.


```shell
docker run --rm -v <local-dir>:<container-dir> -p <local-port>:<container-port> <registry>/<repository>/<image-name>:<tag> bash -c <command>
```

which could results in something like launching a jupyter-notebook and destroying the container after termination:

```shell
docker run --rm -p 8888:8888 ghcr.io/i4ds/karabo-pipeline:latest bash -c 'jupyter lab --ip 0.0.0.0 --no-browser --allow-root --port=8888'
```

This starts a port-forwarded jupyter-lab server in the container, accessible through a browser using the printed URL. If you're operating on a remote server, don't forget to port-forwarding through SSH.

It isn't recommended to just map the uid and gid through the `docker run` command, because for singularity compatibility we can't create a custom user. However, just setting the ID's would result in launched services like `jupyter lab` to lose permissions as root, which is needed to write service-specific files. Therefore, if you intend to add a writable mounted volume, you have to adjust the permissions by yourself.

## Singularity Containers

Singularity containers are often standard on HPC clusters, which do not require special permissions (unlike Docker).
We do not provide ready-made [Singularity containers](https://sylabs.io/). However, they can be easily created from Docker images with the following command (may take a while). You may first have to load the module if it's not available `module load singularity`:

```shell
singularity pull docker://ghcr.io/i4ds/karabo-pipeline
```

This creates a `.sif` file which acts as a singularity image and can be used to launch your application. How to use Singularity containers (e.g. mount directories or enable gpu-support) can be seen in the [Singularity documentation](https://docs.sylabs.io/guides/3.1/user-guide/cli.html). Be aware that Singularity mounts the home-directory by default if start a container from your home-directory, which may or may not be desirable (e.g. `conda init` of your home instead of the container could get executed if you execute a Singularity container interactively). Therefore, for interactive access of a Karabo Singularity container, we suggest to use the `--no-home` flag.

## Sarus Containers

On CSCS, it is recommended to use [Sarus containers](https://sarus.readthedocs.io/en/stable/index.html) (see CSCS [Sarus guide](https://user.cscs.ch/tools/containers/sarus/)). Sarus commands are similar to Docker or Singularity. It is recommended to create a Sarus image in an interactive SLURM job using `srun --pty bash`. 

**Setup**

On daint, you should load `daint-gpu` or `daint-mc` before loading the `sarus` modulefile. We haven't tested the setup on alps. This is supposed to be done at some point.

```shell
module load daint-gpu \# or daint-mc
module load sarus
```

Then you can pull a Docker image to a Sarus image as follows:

```shell
sarus pull ghcr.io/i4ds/karabo-pipeline
```

**MPI (MPICH) Support**

Karabo >= `v0.22.0` supports [MPICH](https://www.mpich.org/)-based MPI processes that enable multi-node workflows on CSCS (or any other system which supports MPICH MPI). Note, on CSCS, mpi-runs are launched through SLURM (not through mpirun or mpiexec) by setting the `-n` (total-mpi-tasks) and `-N` (mpi-tasks-per-node) options when launching a job. So you have to set them according to your task.

```shell
srun -N2 -n2 -C gpu sarus run --mount=type=bind,source=<your_repo>,destination=/workspace ghcr.io/i4ds/karabo-pipeline <mpi_application>
```

Here, an MPI application with 2 processes is launched with your repository mounted in the container. Make sure that you know how many processes are reasonable to run because it can rapidly sum up to a large number of nodehours.

Sarus containers allow native mpi-hook to utilize the mpi of CSCS at optimized performance. This can be done by simply adding the `--mpi` flag to the sarus run command. Probably, there will be some warning about the minor version of some libmpi-files. However, according to [sarus abi-compatibility](https://sarus.readthedocs.io/en/stable/user/abi_compatibility.html) this shouldn't be an issue.