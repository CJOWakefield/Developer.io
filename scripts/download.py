import sys
from pathlib import Path
import asyncio

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.downloader import SatelliteDownloader

async def main():
    downloader = SatelliteDownloader(testing=True)
    await downloader._init_session()
    directory, images = await downloader.process_location(country="United Kingdom",
                                city="London",
                                postcode="SW1A 1AA")
    
    if directory and images:
        downloader.save_to_final_location(directory)
        print(f'\n{len(images)} images downloaded and saved to {directory}')

if __name__ == "__main__":
    asyncio.run(main())
