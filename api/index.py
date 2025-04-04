from flask import Flask, request, jsonify
import os
import json
import google.generativeai as genai
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure API keys
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini if API key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@app.route('/')
def index():
    """Root endpoint."""
    modules = {
        "Flask": True,
        "google.generativeai": bool(GEMINI_API_KEY),
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
        "GEMINI_API_KEY": bool(GEMINI_API_KEY)
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
        
        # Fetch part and system data
        # This is a simplified placeholder for actual data fetching
        compatibility_result = check_compatibility_with_gemini(part_id, system_id)
        
        return jsonify({
            "part_id": part_id,
            "system_id": system_id,
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
        if not GEMINI_API_KEY:
            return {
                "compatible": False,
                "explanation": "Gemini API key is not configured. Cannot perform compatibility check."
            }
            
        # Placeholder for actual Gemini API call
        # In a real implementation, fetch actual part and system details
        # and use Gemini to analyze compatibility
        
        # Simplified placeholder response
        return {
            "compatible": True,
            "explanation": f"Part {part_id} is compatible with System {system_id} based on available specifications."
        }
    
    except Exception as e:
        return {
            "compatible": False,
            "explanation": f"Error during compatibility check: {str(e)}"
        }

# Required for Vercel serverless deployment
if __name__ == "__main__":
    app.run(debug=True)