from src.models import RegionPredictor

def main():
    predictor = RegionPredictor(model_version='v_0_06')
    predictor.predict_region(country="Indonesia",
                             city="Berawa")

if __name__ == "__main__":
    main()
