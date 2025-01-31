from src.data import SatelliteDownloader

def main():
    downloader = SatelliteDownloader()
    downloader.process_location(country="United Kingdom",
                                city="London",
                                postcode="SW1A 1AA")

if __name__ == "__main__":
    main()
