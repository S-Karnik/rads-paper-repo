import hashlib
import os
import ssl
import tarfile
import zipfile
from typing import Optional

import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm

__all__ = [
    "load_parquet",
    "download_from_url",
    "md5_check",
    "extract",
    "load_parquet_file",
]


def load_parquet(
    path: os.PathLike,
) -> pd.DataFrame:
    assert (
        os.path.splitext(path)[1] == ".parquet"
    ), f"File must be a .parquet file, got {path}"
    return pd.read_parquet(path)


__all__ = [
    "disable_ssl_verification",
    "from_url",
    "md5_check",
]


def disable_ssl_verification() -> None:
    """Disable SSL certificate verification.

    This function disables SSL certificate verification for HTTPS requests by setting the default HTTPS context to an unverified one.
    Use with caution as this reduces security.

    ### Example
    ```python
    >>> disable_ssl_verification()
    # SSL verification is now disabled for subsequent HTTPS requests
    ```
    """
    ssl._create_default_https_context = ssl._create_unverified_context


def urlretrieve(
    url: str,
    path: os.PathLike,
    chunk_size: int = 1024 * 32,
    **kwargs,
) -> None:
    """Download a file from a URL and save it to a local path with a progress bar.

    ### Args
        - `url` (str): URL to download from
        - `path` (os.PathLike): Local path to save the downloaded file to
        - `chunk_size` (int): Size of chunks to download at a time. Defaults to 32KB.
        - `**kwargs`: Additional arguments passed to tqdm progress bar

    ### Returns
        - `None`

    ### Example
    ```python
    >>> url = "https://example.com/file.txt"
    >>> path = "downloaded_file.txt"
    >>> urlretrieve(url, path)
    ```
    """
    response = requests.get(url, stream=True, headers={"user-agent": "yupeeee/wah"})
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(path, "wb") as fh, tqdm(total=total_size, **kwargs) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                pbar.update(len(chunk))


def download_from_url(
    url: str,
    root: os.PathLike,
) -> os.PathLike:
    """Download a file from a URL to a local directory.

    ### Args
        - `url` (str): URL to download from
        - `root` (os.PathLike): Local directory to download to

    ### Returns
        - `os.PathLike`: Path to the downloaded file

    ### Example
    ```python
    >>> url = "https://example.com/data.txt"
    >>> root = "data/"
    >>> path = download_url(url, root)
    >>> print(path)
    'data/data.txt'
    ```
    """
    fname = os.path.basename(url)
    path = os.path.join(root, fname)
    # skip download if downloaded
    if os.path.exists(path):
        print(f"{fname} already downloaded to {root}")
    # download if undownloaded
    else:
        os.makedirs(root, exist_ok=True)
        urlretrieve(
            url=url,
            path=path,
            desc=f"Downloading {fname} to {root}",
            chunk_size=1024 * 32,  # 32KB chunks
        )
    return path


def md5_check(
    path: os.PathLike,
    checksum: str,
    chunk_size: int = 1024 * 32,
) -> bool:
    """Check if a file matches an MD5 checksum.

    ### Args
        - `path` (os.PathLike): Path to the file to check
        - `checksum` (str): Expected MD5 checksum to compare against
        - `chunk_size` (int): Size of chunks to read when computing hash. Defaults to 32KB.

    ### Returns
        - `bool`: True if the file's MD5 matches the checksum, False otherwise

    ### Example
    ```python
    >>> path = "data.txt"
    >>> checksum = "d41d8cd98f00b204e9800998ecf8427e"
    >>> md5_check(path, checksum)
    True
    ```
    """
    with open(path, "rb") as f:
        h = hashlib.md5()
        while chunk := f.read(chunk_size):
            h.update(chunk)
        # print(h.hexdigest())
        return h.hexdigest() == checksum


def _set_mode(
    ext: str,
) -> str:
    """Set the mode for opening compressed files.

    ### Args
        - `ext` (str): File extension of the compressed file

    ### Returns
        - `str`: Mode string for opening the compressed file

    ### Example
    ```python
    >>> ext = ".gz"
    >>> mode = _set_mode(ext)
    >>> print(mode)
    'r:gz'
    ```
    """
    modes = {
        ".zip": "r",
        ".tar": "r",
        ".gz": "r:gz",
        ".xz": "r:xz",
    }
    return modes.get(ext, "r")


def extract(
    path: os.PathLike,
    save_dir: Optional[os.PathLike] = None,
) -> None:
    """Extract a compressed file to a directory.

    ### Args
        - `path` (os.PathLike): Path to the compressed file
        - `save_dir` (Optional[os.PathLike]): Directory to extract to. Defaults to the directory containing the compressed file.

    ### Returns
        - `None`

    ### Example
    ```python
    >>> path = "data.zip"
    >>> extract(path)
    # Extracts data.zip to the same directory

    >>> extract(path, "extracted/")
    # Extracts data.zip to extracted/
    ```
    """
    assert os.path.exists(path), f"File {path} does not exist"
    # set extraction mode and save directory
    ext = os.path.splitext(path)[1]
    save_dir = save_dir or os.path.dirname(path)
    if save_dir != os.path.dirname(path):
        os.makedirs(save_dir, exist_ok=True)
    # extract
    try:
        if ext == ".zip":
            with zipfile.ZipFile(path, "r") as f:
                f.extractall(save_dir)
        else:
            mode = _set_mode(ext)
            with tarfile.open(path, mode) as f:
                f.extractall(save_dir)
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        raise ValueError(f"Failed to extract {path}: {str(e)}")

def load_parquet_file(
    fpath: Path,
) -> pd.DataFrame:
    print(fpath)
    assert os.path.exists(fpath) and os.path.basename(fpath).endswith(".parquet")
    return pd.read_parquet(fpath)
