import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from main.py
sys.path.append(str(Path(__file__).parent.parent))

# Import needed modules
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from http import HTTPStatus

# Print environment information
print("Vercel API env-check endpoint loaded")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"GOOGLE_MODEL_NAME: {os.environ.get('GOOGLE_MODEL_NAME', 'not set')}")
print(f"GOOGLE_API_KEY set: {bool(os.environ.get('GOOGLE_API_KEY'))}")
print(f"SUPABASE_URL set: {bool(os.environ.get('SUPABASE_URL'))}")
print(f"SUPABASE_KEY set: {bool(os.environ.get('SUPABASE_KEY'))}")

# Create a simple handler
async def handler(request: Request):
    """
    Handler for the env-check endpoint
    """
    try:
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "SUPABASE_URL_SET": bool(os.environ.get("SUPABASE_URL")),
                "SUPABASE_KEY_SET": bool(os.environ.get("SUPABASE_KEY")),
                "GOOGLE_API_KEY_SET": bool(os.environ.get("GOOGLE_API_KEY")),
                "GOOGLE_MODEL_NAME": os.environ.get("GOOGLE_MODEL_NAME", "not set"),
                "Current ENV": os.environ.get("ENVIRONMENT", "unknown")
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"detail": f"Error in env-check: {str(e)}"}
        ) 