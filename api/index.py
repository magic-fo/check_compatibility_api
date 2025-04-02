import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import from main.py
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

print(f"Python path: {sys.path}")
print(f"Parent directory: {parent_dir}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Files in parent directory: {os.listdir(parent_dir)}")

try:
    # Import app from main.py
    from main import app
    print("Successfully imported app from main.py")
except Exception as e:
    print(f"Error importing app from main.py: {str(e)}")
    import traceback
    traceback.print_exc()

# This file serves as the entry point for Vercel serverless functions 
# Export the FastAPI app for Vercel 