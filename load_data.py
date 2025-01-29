import asyncio
import aiohttp
import aiofiles
import os
import json
from tqdm.asyncio import tqdm
from datasets import load_dataset
from urllib.parse import urlparse
import hashlib

# Configuration
CONCURRENT_DOWNLOADS = 100  # Adjust based on your system and network
SUCCESS_JSON = 'successful_urls.json'
FAILED_JSON = 'failed_urls.json'
IMAGES_DIR = 'downloaded_images'  # Directory to save images

# Ensure the images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)
successed = 0
async def download_image(session, url, semaphore, success_list, failed_list):
    """
    Downloads the image from the given URL and saves it to the images directory.
    If successful, appends a dict with 'url' and 'path' to success_list.
    Otherwise, appends the URL and error/status to failed_list.
    """
    async with semaphore:
        try:
            async with session.get(url, timeout=60) as response:
                if response.status == 200:
                    # Determine the image extension
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' not in content_type:
                        raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")

                    # Extract the file extension
                    ext = ''
                    if 'image/' in content_type:
                        ext = content_type.split('image/')[-1].split(';')[0]
                        if ext == 'jpeg':
                            ext = 'jpg'  # Common extension
                    else:
                        ext = 'jpg'  # Default extension

                    # Create a unique filename using a hash of the URL
                    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
                    filename = f"{url_hash}.{ext}"
                    file_path = os.path.join(IMAGES_DIR, filename)

                    # Read the image data
                    image_data = await response.read()

                    # Save the image asynchronously
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(image_data)

                    # Record the successful download
                    success_list.append({'url': url, 'path': file_path})
                    successed+=1
                    print(successed)
                else:
                    failed_list.append({'url': url, 'status': responsepyth.status})
                    print(responsepyth.status)
        except Exception as e:
            failed_list.append({'url': url, 'error': str(e)})

async def main():
    # Load dataset
    print("Loading dataset...")
    data = load_dataset("allenai/pixmo-pointing", split="train")
    urls = list(set(data['image_url']))  # Ensure unique URLs
    print(f"Total unique URLs to download: {len(urls)}")

    # Initialize semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

    # Lists to hold successful and failed downloads
    success_list = []
    failed_list = []

    # Set up aiohttp ClientSession with connection limits
    connector = aiohttp.TCPConnector(limit=CONCURRENT_DOWNLOADS)
    timeout = aiohttp.ClientTimeout(total=60)  # Total timeout for each request

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create asynchronous tasks for each URL
        tasks = [
            download_image(session, url, semaphore, success_list, failed_list)
            for url in urls
        ]

        # Use tqdm to display progress bar
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading Images"):
            await task

    # Save successful downloads to JSON
    async with aiofiles.open(SUCCESS_JSON, 'w') as json_file:
        await json_file.write(json.dumps(success_list, indent=4))
    print(f"\nSuccessfully downloaded {len(success_list)} images.")
    print(f"List of successful downloads saved to '{SUCCESS_JSON}'.")

    # Save failed downloads to JSON (if any)
    if failed_list:
        async with aiofiles.open(FAILED_JSON, 'w') as json_file:
            await json_file.write(json.dumps(failed_list, indent=4))
        print(f"List of failed downloads saved to '{FAILED_JSON}'.")
    else:
        print("All images downloaded successfully without any failures.")

if __name__ == '__main__':
    asyncio.run(main())