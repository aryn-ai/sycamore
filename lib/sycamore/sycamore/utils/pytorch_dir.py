import os
import sys

# copied from the torch code.  Importing the file that contains it uses 4GB. This uses 100M

# from torch.utils.cpp_extension import _get_build_directory as get_pytorch_build_directory
import torch.version
import torch._appdirs


def get_default_build_root() -> str:
    """
    Return the path to the root folder under which extensions will built.
    For each extension module built, there will be one folder underneath the
    folder returned by this function. For example, if ``p`` is the path
    returned by this function and ``ext`` the name of an extension, the build
    folder for the extension will be ``p/ext``.
    This directory is **user-specific** so that multiple users on the same
    machine won't meet permission issues.
    """
    return os.path.realpath(torch._appdirs.user_cache_dir(appname="torch_extensions"))


def get_pytorch_build_directory(name: str, verbose: bool) -> str:
    root_extensions_directory = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        cu_str = (
            "cpu" if torch.version.cuda is None else f'cu{torch.version.cuda.replace(".", "")}'
        )  # type: ignore[attr-defined]
        python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        build_folder = f"{python_version}_{cu_str}"

        root_extensions_directory = os.path.join(root_extensions_directory, build_folder)

    if verbose:
        print(f"Using {root_extensions_directory} as PyTorch extensions root...", file=sys.stderr)

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            print(f"Creating extension directory {build_directory}...", file=sys.stderr)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    return build_directory
