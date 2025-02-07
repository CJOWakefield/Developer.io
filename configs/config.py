from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

# Get the base directory (where .env file is located)
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

class CloudConfig:
    PROJECT_ID = os.getenv('GCP_PROJECT_ID')
    BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
    SERVICE_ACCOUNT_PATH = f'{BASE_DIR}/service/developerio.json'

    @classmethod
    def validate(cls):
        missing_vars = []
        if not cls.PROJECT_ID:
            missing_vars.append('GCP_PROJECT_ID')
        if not cls.BUCKET_NAME:
            missing_vars.append('GCP_BUCKET_NAME')
        if not cls.SERVICE_ACCOUNT_PATH:
            missing_vars.append('Service Account Key File')

        if missing_vars:
            raise ValueError(f"Missing required configuration: {', '.join(missing_vars)}")

if __name__ == '__main__':
    print(BASE_DIR)
    CloudConfig().validate()
