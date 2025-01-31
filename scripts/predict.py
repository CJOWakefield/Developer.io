from src.models import RegionPredictor

def main():
    predictor = RegionPredictor()
    predictor.predict_region(country="United Kingdom",
                             city="London")

if __name__ == "__main__":
    main()
