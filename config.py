"""Configuration module that loads environment variables from .env file"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Ministral-3B-Instruct-2512")
MODEL_REPO = os.getenv("MODEL_REPO", "https://huggingface.co/mistralai/Ministral-3B-Instruct-2512")

# Validate that required API keys are present
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in .env file. Please set it before running the application.")
