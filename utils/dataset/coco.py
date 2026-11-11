import os

from ..paths import FID_REF_ROOT
from .utils import download_from_url, extract, md5_check

__all__ = [
    "download_coco",
]

URL = "http://images.cocodataset.org/zips/val2017.zip"
CHECKSUM = "442b8da7639aecaf257c1dceb8ba8c80"


def download_coco() -> None:
    path = download_from_url(URL, FID_REF_ROOT)
    while not md5_check(path, CHECKSUM):
        print("Dataset corrupted. Redownloading dataset.")
        os.remove(path)
        path = download_from_url(URL, FID_REF_ROOT)
    extract(path)
    os.remove(path)
