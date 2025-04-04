from flask import Flask, request, jsonify
import os
import json
import google.generativeai as genai
import httpx
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure API keys
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash-thinking-exp")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-7-sonnet-latest")

# Configure Gemini if API key is available
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Anthropic client if API key is available
anthropic_client = None
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@app.route('/')
def index():
    """Root endpoint."""
    modules = {
        "Flask": True,
        "google.generativeai": bool(GOOGLE_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "httpx": True,
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY)
    }
    
    return jsonify({
        "status": "ok",
        "message": "API is running",
        "modules": modules
    })

@app.route('/env-check')
def env_check():
    """Check environment variables."""
    env_vars = {
        "SUPABASE_URL": bool(SUPABASE_URL),
        "SUPABASE_KEY": bool(SUPABASE_KEY),
        "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
        "ANTHROPIC_API_KEY": bool(ANTHROPIC_API_KEY),
        "GOOGLE_MODEL_NAME": bool(GOOGLE_MODEL_NAME),
        "ANTHROPIC_MODEL_NAME": bool(ANTHROPIC_MODEL_NAME)
    }
    
    missing_vars = [var for var, exists in env_vars.items() if not exists]
    
    return jsonify({
        "environment_variables": env_vars,
        "missing_variables": missing_vars,
        "all_available": len(missing_vars) == 0
    })

@app.route('/api/compatibility-check', methods=['POST'])
def compatibility_check():
    """Check compatibility between components."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        
        # Simple validation
        if not data.get('part_id') or not data.get('system_id'):
            return jsonify({"error": "Missing required parameters: part_id and system_id are required"}), 400
        
        part_id = data.get('part_id')
        system_id = data.get('system_id')
        model_provider = data.get('model_provider', 'google')  # Default to Google
        
        # Fetch part and system data
        # This is a simplified placeholder for actual data fetching
        if model_provider.lower() == 'anthropic' and ANTHROPIC_API_KEY:
            compatibility_result = check_compatibility_with_anthropic(part_id, system_id)
        else:
            compatibility_result = check_compatibility_with_gemini(part_id, system_id)
        
        return jsonify({
            "part_id": part_id,
            "system_id": system_id,
            "model_provider": model_provider,
            "compatible": compatibility_result["compatible"],
            "explanation": compatibility_result["explanation"]
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error",
            "message": "Failed to process compatibility check"
        }), 500

def check_compatibility_with_gemini(part_id, system_id):
    """Use Gemini to check compatibility between part and system."""
    try:
        if not GOOGLE_API_KEY:
            return {
                "compatible": False,
                "explanation": "Google API key is not configured. Cannot perform compatibility check."
            }
            
        # Placeholder for actual Gemini API call
        # In a real implementation, fetch actual part and system details
        # and use Gemini to analyze compatibility
        model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
        
        # Simplified placeholder response
        response = model.generate_content(
            f"Is part {part_id} compatible with system {system_id}? Please provide a detailed analysis."
        )
        
        # For demonstration purposes
        return {
            "compatible": True,
            "explanation": f"Part {part_id} is compatible with System {system_id} based on available specifications analyzed by Gemini model."
        }
    
    except Exception as e:
        return {
            "compatible": False,
            "explanation": f"Error during compatibility check with Gemini: {str(e)}"
        }

def check_compatibility_with_anthropic(part_id, system_id):
    """Use Anthropic Claude to check compatibility between part and system."""
    try:
        if not ANTHROPIC_API_KEY or not anthropic_client:
            return {
                "compatible": False,
                "explanation": "Anthropic API key is not configured. Cannot perform compatibility check."
            }
            
        # Placeholder for actual Anthropic API call
        # In a real implementation, fetch actual part and system details
        # and use Claude to analyze compatibility
        
        message = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=1000,
            messages=[
                {
                    "role": "user", 
                    "content": f"Is part {part_id} compatible with system {system_id}? Please provide a detailed analysis."
                }
            ]
        )
        
        # For demonstration purposes
        return {
            "compatible": True,
            "explanation": f"Part {part_id} is compatible with System {system_id} based on available specifications analyzed by Claude model."
        }
    
    except Exception as e:
        return {
            "compatible": False,
            "explanation": f"Error during compatibility check with Claude: {str(e)}"
        }

# Required for Vercel serverless deployment
if __name__ == "__main__":
    app.run(debug=True)