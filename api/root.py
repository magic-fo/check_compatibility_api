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
print("Vercel API root endpoint loaded")

# Create a simple handler for the root endpoint
async def handler(request: Request):
    """
    Handler for the root endpoint
    """
    try:
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={"status": "ok", "message": "API is running"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"detail": f"Error in root endpoint: {str(e)}"}
        ) 