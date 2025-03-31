import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import from main.py
sys.path.append(str(Path(__file__).parent.parent))

# Import app from main.py
from main import app

# This file serves as the entry point for Vercel serverless functions 