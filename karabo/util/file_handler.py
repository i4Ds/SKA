"""
Offers functionality for handling caching data on disk and
reading/writung to temporary files.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from types import TracebackType
from typing import Literal, Optional, Union, overload

from typing_extensions import assert_never

from karabo.util._types import DirPathType, FilePathType, TDirPathType
from karabo.util.helpers import get_rnd_str
from karabo.util.plotting_util import Font
from karabo.warning import _DEV_ERROR_MSG

_LongShortTermType = Literal["long", "short"]


def _get_env_value(
    varname: str,
) -> Union[str, None]:
    """Gets env-var value from OS. Treats empty-str as None.

    Args:
        varname: Varname to get value from.

    Returns:
        Value of `varname`.
    """
    env_value = os.environ.get(varname)
    if env_value is not None and env_value == "":
        env_value = None
    return env_value


def _get_disk_cache_root(
    term: _LongShortTermType,
    create_if_not_exists: bool = True,
) -> str:
    """Gets the root-directory of the disk-cache.

    Args:
        term: Whether to get long- or short-term root.
        create_if_not_exists: Create according dir if not exists?

    Honors 'TMPDIR' & 'TMP' and 'SCRATCH' env-var(s) for STM where
    'TMPDIR' = 'TMP' > 'SCRATCH' > /tmp
    Honors 'XDG_CACHE_HOME' env-var(s) for LTM where
    'XDG_CACHE_HOME' > $HOME/.cache > /tmp
    Note: Setting env-vars has only an effect if they're set before importing Karabo.

    Returns:
        str: Path of tmpdir

    Raises:
        RuntimeError: If 'TMPDIR' & 'TMP' are set differently which is ambiguous.

    Note:
        In a container-setup, this dir is preferably a mounted dir.
        For long-term-memory so that each object doesn't have to be downloaded
        for each run. For short-term-memory so that the created artifacts are
        locatable on the launch system.

        Singularity & Sarus container usually use a mounted /tmp. However, this
        is not the default case for Docker containers. This may be a reason to
        not put the download-objects into /tmp of the Docker-image.

    """
    # first guess is /tmp
    tmpdir = f"{os.path.sep}tmp"
    if term == "short":
        # second guess is if scratch is available (mid prio)
        if (scratch := _get_env_value("SCRATCH")) is not None and os.path.exists(
            scratch
        ):
            tmpdir = scratch
        # third guess is to honor the env-variables mentioned (high prio)
        env_check: Optional[
            str
        ] = None  # variable to check previous environment variables
        environment_varname = ""
        if (TMPDIR := _get_env_value("TMPDIR")) is not None:
            tmpdir = os.path.abspath(TMPDIR)
            env_check = TMPDIR
            environment_varname = "TMPDIR"
        if (TMP := _get_env_value("TMP")) is not None:
            if env_check is not None:
                if TMP != env_check:
                    raise RuntimeError(
                        f"Environment variables collision: TMP={TMP} != "
                        + f"{environment_varname}={env_check} which is ambiguous."
                    )
            else:
                tmpdir = os.path.abspath(TMP)
                env_check = TMP
                environment_varname = "TMP"
    elif term == "long":
        home = _get_env_value("HOME")
        if home is not None:  # should always be set, but just to be sure
            tmpdir = os.path.join(home, ".cache")
        if (xdg_cache_dir := _get_env_value("XDG_CACHE_HOME")) is not None:
            tmpdir = xdg_cache_dir
    else:
        assert_never(term)
    if create_if_not_exists:
        os.makedirs(tmpdir, exist_ok=True)
    return tmpdir


def _get_cache_dir(term: _LongShortTermType) -> str:
    """Creates a user-deterministic cache-dir-name.

    dir-name: karabo-<LTM|STM>-($USER-)<10-rnd-asci-letters-and-digits>

    The random-part of the cache-dir is seeded for relocation purpose.
        Otherwise, the same tmp-dirs couldn't be used in another run.
    The seed prevents tmpdir collisions of different
        users on a cluster.

    Returns:
        cache-dir-name
    """
    delimiter = "-"
    prefix = "karabo"
    if term == "long":
        prefix = delimiter.join((prefix, "LTM"))
    elif term == "short":
        prefix = delimiter.join((prefix, "STM"))
    else:
        assert_never(term)
    user = _get_env_value("USER")
    if user is not None:
        prefix = delimiter.join((prefix, user))
        seed = user + term
    else:
        seed = "42" + term
    suffix = get_rnd_str(k=10, seed=seed)
    cache_dir_name = delimiter.join((prefix, suffix))
    return cache_dir_name


class FileHandler:
    """Utility file-handler for unspecified directories.

    Provides cache-management functionality.
    `FileHandler.root_stm` (short-term-memory-dir) and `FileHandler.root_ltm`
    (long-term-memory-dir) are static root-directories where each according cache-dir
    is located. In case someone wants to extract something specific from the cache,
    the path is usually printed blue & bold in stdout.

    Honors 'TMPDIR' & 'TMP' and 'SCRATCH' env-var(s) for STM-disk-cache where
    'TMPDIR' = 'TMP' > 'SCRATCH' > /tmp
    Honors 'XDG_CACHE_HOME' env-var(s) for LTM-disk-cache where
    'XDG_CACHE_HOME' > $HOME/.cache > /tmp
    Note: Setting env-vars has only an effect if they're set before importing Karabo.
    Run-time adjustments must be done directly on `root_stm` and `root_ltm`!

    The root STM and LTM must be unique per user (seeded rnd chars+digits) to
    avoid conflicting dir-names on any computer with any root-directory.

    LTM-root
    └── karabo-LTM-<user>-<10 rnd chars+digits>
        ├── <prefix><10 rnd chars+digits>
        |    ├── <subdir>
        |    └── <file>
        └── <prefix><10 rnd chars+digits>
             ├── <subdir>
             └── <file>

    STM-root
    └── karabo-STM-<user>-<10 rnd chars+digits>
        ├── <prefix><10 rnd chars+digits>
        |    ├── <subdir>
        |    └── <file>
        └── <prefix><10 rnd chars+digits>
             ├── <subdir>
             └── <file>

    FileHandler can be used the same way as `tempfile.TemporaryDirectory` using `with`.
    """

    root_stm: str = _get_disk_cache_root(term="short")
    root_ltm: str = _get_disk_cache_root(term="long")

    @classmethod
    def ltm(cls) -> str:
        """LTM (long-term-memory) path."""
        return os.path.join(cls.root_ltm, _get_cache_dir(term="long"))

    @classmethod
    def stm(cls) -> str:
        """STM (short-term-memory) path."""
        return os.path.join(cls.root_stm, _get_cache_dir(term="short"))

    def __init__(
        self,
    ) -> None:
        """Creates `FileHandler` instance."""
        # tmps is an instance bound dirs and/or files registry for STM
        self.tmps: list[str] = list()

    @classmethod
    def _get_term_dir(cls, term: _LongShortTermType) -> str:
        if term == "short":
            dir_ = cls.stm()
        elif term == "long":
            dir_ = cls.ltm()
        else:
            assert_never(term)
        return dir_

    @overload
    def get_tmp_dir(
        self,
        prefix: Union[str, None] = None,
        term: Literal["short"] = "short",
        purpose: Union[str, None] = None,
        unique: object = None,
        mkdir: bool = True,
        seed: Optional[Union[str, int, float, bytes]] = None,
    ) -> str:
        ...

    @overload
    def get_tmp_dir(
        self,
        prefix: str,
        term: Literal["long"],
        purpose: Union[str, None] = None,
        unique: object = None,
        mkdir: bool = True,
        seed: Optional[Union[str, int, float, bytes]] = None,
    ) -> str:
        ...

    def get_tmp_dir(
        self,
        prefix: Union[str, None] = None,
        term: _LongShortTermType = "short",
        purpose: Union[str, None] = None,
        unique: object = None,
        mkdir: bool = True,
        seed: Optional[Union[str, int, float, bytes]] = None,
    ) -> str:
        """Gets a tmp-dir path.

        This is the to-go function to get a tmp-dir in the according directory.

        Args:
            prefix: Dir-name prefix for STM (optional) and dir-name for LTM (required).
            term: "short" for STM or "long" for LTM.
            purpose: Creates a verbose print-msg with it's purpose if set.
            unique: If an object which has attributes is provided, then you get
                the same tmp-dir for the unique instance.
            mkdir: Make-dir directly?
            seed: Seed rnd chars+digits of a STM sub-dir for relocation
                purpose of different processes? Shouldn't be used for LTM sub-dirs,
                unless you know what you're doing. LTM sub-dirs are already seeded with
                `prefix`. However, if they are seeded for some reason, the seed is then
                something like `prefix` + `seed`, which leads to different LTM sub-dirs.

        Returns:
            tmp-dir path
        """
        obj_tmp_dir_short_name = "_karabo_tmp_dir_short"
        tmp_dir: Union[str, None] = None
        if unique is not None:
            if term != "short":
                raise RuntimeError(
                    "`unique` not None is just supported for short-term tmp-dirs."
                )
            try:
                unique.__dict__  # just to test try-except AttributeError
                if hasattr(unique, obj_tmp_dir_short_name):
                    tmp_dir = getattr(unique, obj_tmp_dir_short_name)
            except AttributeError as e:
                raise AttributeError(
                    "`unique` must be an object with attributes, "
                    + f"but is of type {type(unique)} instead."
                ) from e

        if tmp_dir is not None:
            dir_path = tmp_dir
            exist_ok = True
        elif term == "short":
            dir_path = FileHandler._get_term_dir(term=term)
            dir_name = get_rnd_str(k=10, seed=seed)
            if prefix is not None:
                dir_name = "".join((prefix, dir_name))
            dir_path = os.path.join(dir_path, dir_name)
            if unique is not None:
                setattr(unique, obj_tmp_dir_short_name, dir_path)
            self.tmps.append(dir_path)
            if seed is None:
                exist_ok = False
            else:
                exist_ok = True
        elif term == "long":
            dir_path = FileHandler._get_term_dir(term=term)
            if prefix is None:
                raise RuntimeError(
                    "For long-term-memory, `prefix` must be set to have unique dirs."
                )
            if seed is not None:
                seed = prefix + str(seed)
            else:
                seed = prefix
            dir_name = get_rnd_str(k=10, seed=seed)
            dir_name = "".join((prefix, dir_name))
            dir_path = os.path.join(dir_path, dir_name)
            exist_ok = True
        else:
            assert_never(term)
        if not exist_ok and os.path.exists(dir_path):
            raise FileExistsError(f"{dir_path} already exists")
        if mkdir and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=exist_ok)
            if purpose and len(purpose) > 0:
                purpose = f" for {purpose}"
                print(f"Creating {Font.BLUE}{Font.BOLD}{dir_path}{Font.END}{purpose}")
        return dir_path

    def clean_instance(self) -> None:
        """Cleans instance-bound tmp-dirs of `self.tmps` from disk."""
        tmps = copy(self.tmps)
        for tmp in tmps:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            self.tmps.remove(tmp)

    @classmethod
    def clean(
        cls,
        term: _LongShortTermType = "short",
    ) -> None:
        """Removes the entire directory specified by `term`.

        We strongly suggest to NOT use this function in a workflow. This function
        removed the entire karabo-disk-cache. So if there's another karabo-process
        running in parallel, you could mess with their disk-cache as well.

        Args:
            term: "long" or "short" term memory
        """
        dir_ = cls._get_term_dir(term=term)
        if os.path.exists(dir_):
            shutil.rmtree(dir_)

    @classmethod
    def is_dir_empty(cls, dirname: DirPathType) -> bool:
        """Checks if `dirname` is empty assuming `dirname` exists.

        Args:
            dirname: Directory to check.

        Raises:
            NotADirectoryError: If `dirname` is not an existing directory.

        Returns:
            True if dir is empty, else False
        """
        if not os.path.isdir(dirname):
            raise NotADirectoryError(f"{dirname} is not an existing directory.")
        is_empty = len(os.listdir(path=dirname)) == 0
        return is_empty

    @classmethod
    def remove_empty_dirs(cls, term: _LongShortTermType = "short") -> None:
        """Removes empty directories in the chosen cache-dir.

        Args:
            term: "long" or "short" term memory
        """
        dir_ = _get_cache_dir(term=term)
        paths = glob.glob(os.path.join(dir_, "*"), recursive=False)
        for path in paths:
            if os.path.isdir(path) and len(os.listdir(path=path)) == 0:
                shutil.rmtree(path=path)

    @classmethod
    def empty_dir(cls, dir_path: DirPathType) -> None:
        """Deletes all contents of `dir_path`, but not the directory itself.

        This function assumes that all files and directories are owned by
        the function-user.

        Args:
            dir_path: Directory to empty.
        """
        shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=False)

    def __enter__(self) -> str:
        return self.get_tmp_dir(prefix=None, term="short")

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.clean_instance()


def assert_valid_ending(
    path: Union[str, FilePathType, DirPathType], ending: str
) -> None:
    """Utility function to check if the ending of `path` is `ending`.

    Args:
        path: Path to check.
        ending: Ending match.

    Raises:
        ValueError: When the ending of `path` doesn't match `ending`.
    """
    path_ = str(path)
    if not path_.endswith(ending):
        fname = path_.split(os.path.sep)[-1]
        raise AssertionError(
            f"Invalid file-ending, file {fname} must have {ending} extension!"
        )


def getsize(inode: Union[str, Path]) -> int:
    """Gets the total size of a file or directory in number of bytes.

    Args:
        inode: Directory or file to get size from. Can take a while for a large dir.

    Returns:
        Number of bytes of `inode`.
    """
    inode_path = Path(inode)
    if not inode_path.exists():  # check validity before passing to system-call
        err_msg = f"{inode=} doesn't exist!"
        raise RuntimeError(err_msg)
    if os.path.isdir(inode_path):
        try:
            du_out = subprocess.run(  # sh should be supported by any linux/WSL dist
                ["du", "-sb", str(inode_path)], check=True, capture_output=True
            )
            nbytes = int(du_out.stdout.decode().split(sep="\t")[0])
        except Exception as e:
            err_msg = (
                f"Get size of {inode=} failed unexpectedly (most likely a dev-error)."
            )
            raise RuntimeError(err_msg) from e
    else:
        nbytes = os.path.getsize(filename=inode)
    return nbytes


@contextmanager
def write_dir(
    dir: TDirPathType, *, overwrite: bool = False
) -> Generator[TDirPathType, None, None]:
    """Enables transactional creating and writing into `dir`.

    Assumes that `dir` is empty, meaning NOT partially filled with anything.

    This function is NOT thread-safe!

    Args:
        dir: Directory to create & write.
        overwrite: Allow overwrite? Be aware that it will replace the entire
            `dir` if it already exists. So be careful to provide the correct dir.

    Yields:
        Directory to fill safely.
    """
    dir_path = Path(dir)  # `Path(dir)` handles trailing "/"
    parent_dir, dirname = os.path.split(dir_path)
    if not overwrite and os.path.exists(dir_path):
        err_msg = f"{dir} already exists."
        raise FileExistsError(err_msg)
    new_dirs = [
        parent for parent in [dir_path, *dir_path.parents] if not os.path.exists(parent)
    ]
    root_created_dir = new_dirs[-1] if len(new_dirs) > 0 else None
    if root_created_dir is not None and os.path.exists(root_created_dir):
        raise RuntimeError(_DEV_ERROR_MSG)  # to absolutely be sure it doesn't exist
    rnd_str = get_rnd_str(k=10, seed=dirname)
    tmpdir = type(dir)(os.path.join(parent_dir, f".{rnd_str}-{dirname}"))
    tmpdir_old = os.path.join(parent_dir, f".{rnd_str}-old-{dirname}")
    try:
        os.makedirs(tmpdir, exist_ok=False)
        yield tmpdir  # type: ignore[misc]
        if overwrite and os.path.exists(dir_path):
            os.rename(src=dir_path, dst=tmpdir_old)
            os.rename(src=tmpdir, dst=dir_path)
            shutil.rmtree(
                path=tmpdir_old
            )  # potentially dangerous if `dir` is accidentally set to another dir
        else:
            os.rename(src=tmpdir, dst=dir_path)
    except BaseException:
        if os.path.exists(tmpdir_old):
            shutil.rmtree(path=tmpdir_old)
        if os.path.exists(tmpdir):
            shutil.rmtree(path=tmpdir)
        if root_created_dir is not None and os.path.exists(root_created_dir):
            shutil.rmtree(path=root_created_dir)
        raise
