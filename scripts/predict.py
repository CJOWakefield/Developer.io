from src.models import RegionPredictor

''' ----- Predict script summary -----

> main - Run prediction on a specific region using the RegionPredictor class.
    >> Inputs: None (hardcoded values)
    >> Process: Creates RegionPredictor instance with model v_0_06 and predicts land types for Andalusia, Spain
    >> Outputs: Saves predicted masks to data/downloaded directory
'''

def main():
    predictor = RegionPredictor(model_version='v_0_06')
    predictor.predict_region(country="Spain",
                             city="Andalusia")
                             


if __name__ == "__main__":
    main()

