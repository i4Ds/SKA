# Development Documentation

## Setup local dev environment

Make sure you have a working installation of Conda, with, e.g.:

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
```

(Optional but recommended for performance) Set the solver to libmamba:

```shell
conda config --set solver libmamba
```

Next, clone the code via git, and cd into it:
```shell
git clone git@github.com:i4Ds/Karabo-Pipeline.git
cd Karabo-Pipeline
```

Then create a local development environment with the provided `environment.yaml` file.

```shell
conda env create -n <your-env-name> -f environment.yaml
conda activate <your-env-name>
```

In case you're using WSL2, do the following (see [issue 550](https://github.com/i4Ds/Karabo-Pipeline/issues/550)):
```shell
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib
```

Then install Karabo as a package and the according dev-dependencies.

```shell
pip install -e ".[dev]"
```

Afterwards, activating you dev-tools in your IDE and SHELL is recommended. For the setup of your IDE of choice you have to do it yourself. For the SHELL setup, we recommend to do the following in the repo-root:

```shell
pre-commit install
podmena add local
```

## Formatting

To increase the readability of the code and to better detect potential errors already during development, a number of tools are used. The configurations of the tools are handled in `setup.cfg` or `pyproject.toml`. If changes to the configurations are desired, the team members should agree to this (e.g. via a meeting).

It is possible that certain tools complain about something that is not easy or even impossible to fix. ONLY then, there are options to ignore certain lines of code or even whole files for the checker. E.g. `# noqa` ignores inline flake8 complaints. But be careful not to accidentally ignore the whole file (e.g. with `# flake8: noqa`). Please refer to the documentation of the respective tool to learn how to ignore the errors.

We recommend the setup using [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks), so that the in `.pre-commit-config.yaml` specified tools prevent commits if the checks on the added files fail. The hook can be easily initialized using `pre-commit install` in the root directory of the repository. If absolutely necessary, the hooks can be ignored using the `--no-verify` flag in the git-commands. However, some checks are included in the CI-tests, so you have to check your files anyway. The [pre-commit-configs](https://pre-commit.com/) used for this are defined in `.pre-commit-config.yaml`.

The following sections list the tools to be used and their function & usage.

### black

[https://github.com/psf/black](black) is a Python [PEP 8](https://peps.python.org/pep-0008/) compliant opinionated code formatter. It formats entire Python files with the CLI command `black {source_file_or_directory}`. For specific options, see [command line options](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) or `black --help`.

### isort

[isort](https://pycqa.github.io/isort/) organizes Python imports alphabetically and automatically separated into sections and by type. The command can be called via `isort {source_file}`. There are further possibilities in the [documentation](https://pycqa.github.io/isort/) to apply isort to multiple files at once.

### mypy

[mypy](https://mypy.readthedocs.io/en/stable/) is a static type checker for Python. It ensures that variables and functions are used correctly in the code. It warns the developer if type hints ([PEP 484](https://peps.python.org/pep-0484/)) are not used correctly. The CLI `mypy {source_file_or_directory}` makes it easy to use mypy (although it may take a while). For specific options, see [command line options](https://mypy.readthedocs.io/en/stable/command_line.html) or `mypy --help`. The configs set in `setup.cfg` are `--strict`, so there is no need to set the flag by yourself.

If you run mypy several times for the same file(s) and it takes a while, you can launch the mypy daemon `dmypy start` and then check your file(s) using `dmypy check {file_or_directory}`. The first time you check might still take a while. However, once the check is done and you want to check a file again, the check will only take into account the changes you have made, so it will be much faster.

### pydocstyle

[pydocstyle](http://www.pydocstyle.org/en/stable/) supports docstring checking out of the box. The check can be used via CLI `pydocstyle {source_file_or_directory}`. Further options can be found in the [documentation](http://www.pydocstyle.org/en/stable/usage.html) or via `pydocstyle --help`.

In our project we use the [google convention](http://www.pydocstyle.org/en/stable/error_codes.html?highlight=google#default-conventions). Python files, classes as well as functions need corresponding docstrings. There are autocompletion tools for IDE's that create docstrings in the correct format (e.g. VSCode [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)). Below is an example of a dummy function with the corresponding docstring template:

```
def fun(arg1: int, arg2: int) -> int:
    """__summary__

    __detailed_description_if_needed__

    Args:
        some_arg: __description__

    Returns:
        __description__
    """
    arg_added = arg1 + arg2
    return arg_added
```

### flake8

[flake8](https://flake8.pycqa.org/en/latest/) is a tool for style guide enforcement. It checks various style errors such as unused import, variables, maximum line length, spacing and others. The style-checker can be used via the CLI `flake8 {source_file_or_directory}`. In addition, you can set your IDE's **Linter to flake8**, so that the errors are displayed directly (it might be necessary to install an extension to do so).

It is recommended to run *black* before manually checking with *flake8* because *black* might fix a lot of *flake8* related issues. It is possible that the settings of *isort* and *black* are not compatible with *flake8* and therefore should get changed.

## Add Data Sets by Uploading to CSCS

Make new datasets available through CSCS object-storage REST-API. The GET-request URI follows the following format: `{CSCS-object-store-prefix}/{container}/{object}` where the prefix comes from [CSCS API access](https://castor.cscs.ch/dashboard/project/api_access/).

### Upload through Web Interface

1. Go to [Castor](https://castor.cscs.ch/) and authenticate yourself.
2. Navigate to `project/object-storage/container` and choose the container you want to upload (e.g. `karabo_public`). 
3. Click the upload option and upload the file of choice.

### Upload Through CLI
#### Setup
Read https://user.cscs.ch/tools/openstack/ and implement everything inside `OpenStack CLI Access via Virtual Environment:`

#### Upload Single File
Upload a single file with `swift upload karabo_public <file>`

#### Upload Multiple files
1. Create a new folder inside the container with `swift post karabo_public <folder_name>`
2. Upload all files inside the folder with `swift upload karabo_public <folder_name>`

## Add SKA layouts as OSKAR telescopes

In karabo.data._add_oskar_ska_layouts, there is the script array_config_to_oskar_tm.py to convert an ska-ost-array-config array layout to an OSKAR telescope model (.tm directory), which can then be added to Karabo by copying it to karabo/data/ and adding the telescope and telescope version to karabo/simulation/telescope.py and karabo/simulation/telescope_versions.py, respectively. I decided to add each layout as a telescope, e.g. SKA-LOW-AA0.5, and use ska-ost-array-config-\[SKA_OST_ARRAY_CONFIG_PACKAGE_VERSION\] as the version, e.g. ska-ost-array-config-2.3.1.

Code to be adjusted:
- telescope.py: add telescope to OSKARTelescopesWithVersionType
- telescope.py: add telescope to directory name mapping to OSKAR_TELESCOPE_TO_FILENAMES
- telescope_versions.py: add versions enum, see existing enums for inspiration
- telescope.py: add telescope to versions class mapping to OSKAR_TELESCOPE_TO_VERSIONS

Setup info to run the script can be found in the README in karabo.data._add_oskar_ska_layouts.

## Add ALMA layouts as OSKAR telescopes

The dishes of the ALMA telescope (Atacama Large Microwave Array) can operate in different configurations. These 'cycles' are set up at different points in time (see link [1]). For the next year (2025) three new cycles are added to the configuration: cycle 9, 10 and 11. There is a script that helps to import new cycle configurations to Karabo. It fetches the latest config files from the ALMA server (see link [2]) and converts them to an OSKAR telescope model (.tm directory). This can then be added to Karabo by copying it to `karabo/data/`. 

The script and a README file can be found in `karabo.data._add_oskar_alma_layouts`.

The files are fetched directly from the ALMA server. The url is not under our control. Therefore, the url may change which breaks the code. In this case have a look at link [2] and update the variable `CONFIG_FILE_URL` in the code.

###  Setup
1. Create and activate a Python virtual env / conda env or similar
2. `pip install requests` should be enough
3. Adjust code line `ALMA_CYCLE = 10` according to the desired version of the ALMA cycle.

### Convert the configurations
1. Run the script. The OSKAR .tm-foders are written to the current working directory
2. Copy the directories to  `karabo/data/`
3. Add the cycle name and version to `karabo/simulation/telescope.py` and `karabo/simulation/telescope_versions.py`, respectively.

### Important links
1. The ALMA configuration schedule: https://almascience.eso.org/news/the-alma-configuration-schedule-for-cycles-9-10-and-11
2. The configuration files: https://almascience.eso.org/tools/casa-simulator


## Update Documentation

The docs are built from the python source code and other doc source files located in /doc/src.
The .rst and .md files need to be referenced inside of index.rst or an already referenced page inside of index.rst, in order to be viewable by the public upon building the documentation.

If you wish to add a new documentation file, create a new .md file or .rst file inside of the /doc/src folder.
Then, reference the new file within index.rst in the topmost toctree. For instance, see below:

````rst


Welcome to Karabo-Pipeline's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Container
   modules
   examples/Examples.md
   --> ADD THE REFERENCE TO YOUR FILE HERE


````

(Note that .md files can reference each other, e.g. with ``[some file](path/to/some/file)``).

Once you have made changes to the documentation, you can test them via the following steps:

```shell
# Inserts code snippets into Examples documentation
python doc/src/examples/combine_examples.py
cp -a doc/src/ _build
sphinx-apidoc . -o _build
make html
```

Then, you can serve the documentation locally:

```shell
cd _deploy/html
python -m http.server 3000
```

If you add new examples to karabo/examples, please make sure to reference them on the examples page in the docs with a short description so they're visible to users. Edit example_structure.md and then generate examples.md using combine_examples.py.

## Update Tests

We use the ` pytest` python package ([pytest docs](https://docs.pytest.org/)), with a few imports from the `unittest` package ([unittest docs](https://docs.python.org/3/library/unittest.html)). To add a new test simply go to the `karabo/test` folder.

Add tests for when you write some sort of new code that you feel like might break. Be aware that tests utilize the functionality of the testing-framework and therefore might not behave exactly the same as you would execute the code just as a function. The most important file to consider is `conftest.py`, which could impact the other tests.

## Create a Release
When everything is merged which should be merged, a new Release can be deployed as following:
- [Karabo-Pipeline | Releases](https://github.com/i4Ds/Karabo-Pipeline/releases)
- Click on `Draft a new release`
- Define a Version by clicking `Choose a tag`. We follow PEP440 {major}.{minor}.{path} with a leading `v` at the beginning (see previous versions). Usually we increment the minor version by 1.
- Check that the `Target` is set to `main`.
- Describe the release (get inspired by the previous releases).
- Click `Publish release`. 
- Check on [Karabo-Pipeline | Github Actions](https://github.com/i4Ds/Karabo-Pipeline/actions) that the release is successful. 
- Check that the new version is on [Anaconda.org | Packages](https://anaconda.org/i4ds/karabo-pipeline)
- Check on [Karabo-Pipeline | Docker Images](https://github.com/i4ds/Karabo-Pipeline/pkgs/container/karabo-pipeline) that the released image is live.

