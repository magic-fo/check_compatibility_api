import os
import json
import google.generativeai as genai

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini with environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.environ.get("GOOGLE_MODEL_NAME", "gemini-2.0-flash-thinking-exp")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY is not set in .env file")
    exit(1)

print(f"Using model: {GOOGLE_MODEL_NAME}")
print(f"API key loaded: {'Yes' if GOOGLE_API_KEY else 'No'}")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

try:
    # List available models
    models = genai.list_models()
    print("\nAvailable models:")
    for model in models:
        print(f" - {model.name}")
    
    # Try to create the model
    model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
    print(f"\nSuccessfully created model: {GOOGLE_MODEL_NAME}")
    
    # Test a simple prompt
    prompt = "Write a short poem about drones in 4 lines."
    print(f"\nSending test prompt: '{prompt}'")
    
    response = model.generate_content(prompt)
    print("\nResponse:")
    print(response.text)
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc() 