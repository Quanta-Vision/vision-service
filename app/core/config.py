# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "aws")
MONGODB_URL = os.getenv("MONGODB_URL")
PORT = int(os.getenv("PORT", 8000))
