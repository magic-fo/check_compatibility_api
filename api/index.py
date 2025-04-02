import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import from main.py
sys.path.append(str(Path(__file__).parent.parent))

# Print environment information
print("Vercel API directory loaded")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Environment variables:")
print(f"  GOOGLE_MODEL_NAME: {os.environ.get('GOOGLE_MODEL_NAME', 'not set')}")
print(f"  GOOGLE_API_KEY set: {bool(os.environ.get('GOOGLE_API_KEY'))}")
print(f"  SUPABASE_URL set: {bool(os.environ.get('SUPABASE_URL'))}")
print(f"  SUPABASE_KEY set: {bool(os.environ.get('SUPABASE_KEY'))}")

# Import app from main.py
from main import app

# This file serves as the entry point for Vercel serverless functions 