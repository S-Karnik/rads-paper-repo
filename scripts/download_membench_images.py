import pandas as pd
import requests
import os
import ast
from tqdm import tqdm

import pandas as pd
import requests
import os
import ast
from tqdm import tqdm
import concurrent.futures

def download_single_image(args):
    """
    Helper function to download a single image.
    args: (index, url_index, url, output_dir)
    """
    index, i, url, output_dir = args
    if not url or not isinstance(url, str):
        return False

    try:
        # Determine file extension (basic method)
        ext = os.path.splitext(url)[1].split('?')[0] # Remove query params
        if not ext:
            ext = '.jpg' # Default to jpg if no extension found
        
        filename = f"{index}_{i}{ext}"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
             return True

        # Use a session for potentially better performance if we were reusing connections, 
        # but here domains vary wildly so simple get is fine.
        # Enforce a strict timeout for both connect and read.
        response = requests.get(url, timeout=(5, 10)) 
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return True

    except Exception:
        # Silently fail or log debug if needed, but for bulk download we just want to count errors
        return False

def download_images(csv_path, output_dir, max_workers=20):
    """
    Parses the CSV file and downloads images from URLs into the output directory using parallel threads.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    if 'urls' not in df.columns:
        print(f"Error: 'urls' column not found in: {df.columns}")
        return

    print(f"Found {len(df)} rows in CSV. Preparing tasks...")

    tasks = []
    
    # Pre-process rows to create a list of download tasks
    for index, row in df.iterrows():
        urls_str = row['urls']
        try:
            urls = ast.literal_eval(urls_str)
        except (ValueError, SyntaxError):
            continue

        if isinstance(urls, list):
            for i, url in enumerate(urls):
                tasks.append((index, i, url, output_dir))

    total_images = len(tasks)
    print(f"Found {total_images} total image URLs to process.")
    
    downloaded_count = 0
    error_count = 0

    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to the executor
        # We wrap in tqdm for a progress bar
        results = list(tqdm(executor.map(download_single_image, tasks), total=total_images, desc="Downloading Images"))

    downloaded_count = sum(results)
    error_count = total_images - downloaded_count
    
    print(f"Download complete.")
    print(f"Total processed: {total_images}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    csv_file = "SD1_final.csv"
    output_folder = "membench_images"
    download_images(csv_file, output_folder)
