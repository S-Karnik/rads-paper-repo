"""
A Reproducible Extraction of Training Images from Diffusion Models
Ryan Webster
arXiv (2023)

Paper: https://arxiv.org/abs/2305.08694
GitHub: https://github.com/ryanwebster90/onestep-extraction
"""

import os
from io import BytesIO

import requests
import tqdm
import urllib3
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import utils

if __name__ == "__main__":
    # Suppress insecure request warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    for dataset_name in [
        "SDV1",
        "RealVis",
    ]:
        # Download/load prompts
        dataset = utils.load_prompts(dataset_name)

        # Create image directory
        image_dir = os.path.join(
            utils.paths.DATASET_DIR,
            dataset_name,
            "image"
        )
        os.makedirs(image_dir, exist_ok=True)

        # Setup robust session
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )

        success_count = 0
        failure_count = 0

        for idx, url, _, _ in tqdm.tqdm(
            dataset,
            desc=f"Downloading images ({dataset_name})",
            total=len(dataset),
        ):
            image_path = os.path.join(image_dir, f"{idx}.png")
            # Skip if already downloaded
            if os.path.exists(image_path):
                success_count += 1
                continue

            try:
                # Timeout of 15 seconds for connect, 60 for read
                response = session.get(url, timeout=(15, 60), verify=False)
                if response.status_code == 200:
                    try:
                        # Try to open the image to check validity
                        img = Image.open(BytesIO(response.content))
                        img.verify()
                        # If valid, save the image
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        success_count += 1
                    except Exception as e:
                        print(f"Invalid image: {url}")
                        print(e)
                        # Invalid image content
                        failure_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                print(f"Failed to download image: {url}")
                print(e)
                # Network/connection error
                failure_count += 1

        assert success_count + failure_count == len(dataset), "How did this happen?"
        print(
            f"[{dataset_name}] Downloaded {success_count}/{len(dataset)} images to {image_dir}"
        )
