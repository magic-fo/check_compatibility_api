import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Part Compatibility API")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Google Gemini client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash-thinking-exp")

# --- Data Models ---
class CompatibilityCheckRequest(BaseModel):
    """Request model for compatibility check endpoint"""
    part_ids: List[str] = Field(..., description="List of part IDs to check compatibility between")

class CompatibilityCheckResponse(BaseModel):
    """Response model for a single part compatibility result"""
    part_id: str = Field(..., description="Part ID")
    product_name: str = Field(..., description="Product name")
    available: bool = Field(..., description="Whether the part is compatible with other parts")
    incompatibility: Optional[Dict[str, str]] = Field(None, description="Incompatibility details if available is false")

# --- Data Collection Functions ---
def get_parts_info(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get information about specified parts
    
    Args:
        part_ids: List of part IDs to retrieve
        
    Returns:
        List of part information dictionaries
        
    Raises:
        HTTPException: If parts cannot be found
    """
    if not part_ids:
        return []
        
    response = supabase.table("parts").select("*").in_("id", part_ids).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail=f"No parts found with IDs: {part_ids}")
    
    return response.data

def get_subsystems_for_parts(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get subsystems that contain the specified parts
    
    Args:
        part_ids: List of part IDs
        
    Returns:
        List of subsystem information dictionaries
    """
    if not part_ids:
        return []
    
    # This query finds subsystems where part_ids array contains any of the specified part_ids
    subsystems = []
    
    for part_id in part_ids:
        try:
            # part_ids in subsystems is numeric array, so convert the string ID to numeric
            numeric_part_id = float(part_id)
            response = supabase.table("subsystems").select("*").contains("part_ids", [numeric_part_id]).execute()
            if response.data:
                for subsystem in response.data:
                    # Avoid duplicate subsystems
                    if not any(s["id"] == subsystem["id"] for s in subsystems):
                        subsystems.append(subsystem)
        except ValueError:
            # Skip if part_id is not a valid number
            continue
    
    return subsystems

def get_systems_for_subsystems(subsystem_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get systems that contain the specified subsystems
    
    Args:
        subsystem_ids: List of subsystem IDs
        
    Returns:
        List of system information dictionaries
    """
    if not subsystem_ids:
        return []
    
    systems = []
    
    for subsystem_id in subsystem_ids:
        # subsystem_ids in systems table are stored as strings within the array
        response = supabase.table("systems").select("*").contains("subsystem_ids", [subsystem_id]).execute()
        if response.data:
            for system in response.data:
                # Avoid duplicate systems
                if not any(s["id"] == system["id"] for s in systems):
                    systems.append(system)
    
    return systems

def get_subsystems_and_systems_for_parts(part_ids: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Get subsystems and systems related to the specified parts
    
    Args:
        part_ids: List of part IDs
        
    Returns:
        Tuple of (subsystems, systems)
    """
    subsystems = get_subsystems_for_parts(part_ids)
    subsystem_ids = [s["id"] for s in subsystems]
    systems = get_systems_for_subsystems(subsystem_ids)
    
    return subsystems, systems

# --- LLM Integration Functions ---
def format_llm_input(
    parts: List[Dict[str, Any]], 
    subsystems: List[Dict[str, Any]], 
    systems: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Format part information for LLM in the specified field order
    
    Args:
        parts: List of part information dictionaries
        subsystems: List of subsystem information dictionaries
        systems: List of system information dictionaries
        
    Returns:
        List of formatted part information dictionaries with fields in the specified order
    """
    formatted_parts = []
    
    # Create mapping from part ID to subsystem
    part_to_subsystem = {}
    for subsystem in subsystems:
        subsystem_id = subsystem["id"]
        part_ids = subsystem.get("part_ids", [])
        for part_id in part_ids:
            # Convert numeric part_id to string for consistent key type
            part_to_subsystem[str(part_id)] = subsystem
    
    # Create mapping from subsystem ID to system
    subsystem_to_system = {}
    for system in systems:
        system_id = system["id"]
        subsystem_ids = system.get("subsystem_ids", [])
        for subsystem_id in subsystem_ids:
            # subsystem_ids are already strings in the system table
            subsystem_to_system[subsystem_id] = system
    
    for part in parts:
        part_id = part["id"]
        # Use string part_id for lookup since we standardized the mapping keys to strings
        subsystem = part_to_subsystem.get(part_id)
        if not subsystem:
            # Try again with numeric conversion (to handle case where IDs might be stored as numbers)
            try:
                numeric_id = float(part_id)
                subsystem = part_to_subsystem.get(str(numeric_id))
            except:
                pass
        
        subsystem_id = subsystem["id"] if subsystem else None
        system = subsystem_to_system.get(subsystem_id) if subsystem_id else None
        
        # Format part info in the specified field order
        formatted_part = {
            "system_description": system.get("system_description", "") if system else "",
            "subsystem_name": subsystem.get("subsystem_name", "") if subsystem else "",
            "subsystem_description": subsystem.get("subsystem_description", "") if subsystem else "",
            "part_name": part.get("part_name", ""),
            "product_name": part.get("product_name", ""),
            "part_id": part_id,
            "part_description": part.get("part_description", ""),
            "specifications": part.get("specifications", {}),
            "dimensions": part.get("dimensions", {}),
            "weight": part.get("weight", {})
        }
        
        formatted_parts.append(formatted_part)
    
    return formatted_parts

def check_compatibility_with_llm(formatted_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Gemini LLM to evaluate compatibility between parts
    
    Args:
        formatted_parts: List of formatted part information dictionaries
        
    Returns:
        List of compatibility results
        
    Raises:
        HTTPException: If the LLM API call fails
    """
    # System instruction for the LLM
    system_instruction = """
    Goal:
    You must evaluate and verify the compatibility of mechanical parts with other parts in the subsystem using their detailed specifications, dimensions, and attributes.

    Criteria for Compatibility and Interchangeability Inspection Between Parts:
    - Physical Interface Compatibility
        - Dimensional Verification: Confirm critical dimensions and tolerance matching.
        - Mechanical Connections: Ensure exact matching of threads, keyways, splines, flanges, etc.
        - Mounting Structure: Verify that mounting hole positions, diameters, and patterns match exactly.
    - Functional Compatibility
        - Performance Parameter Verification: Compare key performance indicators such as torque, output, flow rate, pressure, etc.
        - Operating Range Compatibility: Confirm compatibility of RPM, load range, pressure range, etc.
        - System Responsiveness: Verify that response time and acceleration/deceleration characteristics match.
        - Operating Characteristics: Evaluate the impact of vibration, noise, and heat generation on overall system performance.
    - Electrical/Electronic Compatibility
        - Electrical Specifications: Match voltage, current, impedance, and frequency requirements.
        - Connector Compatibility: Verify matching of pin layout, connector type, and size.
        - Signal Interface: Ensure communication protocols and signal levels are compatible.
        - EMI/EMC Characteristics: Evaluate electromagnetic interference generation and immunity.
    - Material and Environmental Compatibility
        - Thermal Expansion Characteristics: Predict issues caused by differences in thermal expansion coefficients between materials.
        - Temperature Influence Zone: Assess the thermal impact of heat-generating components on surrounding parts.
    - Practical Verification Methods
        - Durability Testing: Verify compatibility issues such as wear and fatigue during long-term use.
        - Boundary Condition Testing: Verify performance under extreme conditions including maximum/minimum loads, temperatures, speeds, etc.
    """
    
    # User prompt with formatted parts
    prompt = f"""
    Please evaluate the compatibility between the following parts:
    
    Parts:
    {json.dumps(formatted_parts, indent=2)}
    
    Return a JSON array with one object per part, where each object contains:
    - part_id: The part ID as a string (exactly as provided in the input)
    - product_name: The product_name string of the part
    - available: A boolean indicating whether the part is compatible with other parts
    - incompatibility: If available is true, set this to null. If available is false, provide an object where each key is an incompatible part's ID (as a string) and each value is a string explaining the reason for incompatibility
    
    IMPORTANT: Always preserve the exact format of part_id as provided in the input. Do not change it to a number, keep it as a string.
    
    Ensure that a part's available status is set to false ONLY when it has a direct incompatibility with another part.
    """
    
    # Configure Gemini model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "response_mime_type": "application/json",
    }
    
    try:
        # Initialize Gemini model with system instruction
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Parse response text as JSON
        try:
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from markdown if wrapped in code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response.text, re.MULTILINE)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If still not valid JSON, raise exception
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse LLM response as JSON. Response: {response.text[:500]}..."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error calling Gemini API: {str(e)}"
        )

# --- Compatibility Graph Functions ---
def extract_compatibility_edges(
    llm_response: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract compatibility graph edges from LLM response
    
    Args:
        llm_response: LLM response containing compatibility results
        
    Returns:
        List of compatibility graph edges
    """
    edges = []
    processed_pairs = set()  # Track processed part pairs to avoid duplicates
    
    for part_result in llm_response:
        part_id = part_result.get("part_id")
        if not part_id or part_result.get("available", True):
            continue  # Skip compatible parts
        
        incompatibility = part_result.get("incompatibility", {})
        if not incompatibility:
            continue
        
        for incompatible_part_id_str, reason in incompatibility.items():
            try:
                # 이미 문자열이므로 변환할 필요 없음
                incompatible_part_id = incompatible_part_id_str
                
                # Sort part IDs to ensure consistent edge representation
                part_ids = sorted([part_id, incompatible_part_id])
                pair_key = f"{part_ids[0]}_{part_ids[1]}"
                
                # Skip if this pair has already been processed
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Create edge
                edge = {
                    "part_ids": part_ids,
                    "reason": reason
                }
                
                edges.append(edge)
                
            except Exception:
                # Skip invalid part IDs
                continue
    
    return edges

def update_system_compatibility_graph(system_id: str, new_edges: List[Dict[str, Any]]) -> None:
    """
    Update the compatibility graph in a system record
    
    Args:
        system_id: ID of the system to update
        new_edges: New compatibility edges to add
        
    Raises:
        HTTPException: If the system cannot be found or updated
    """
    # Get current compatibility graph
    system_response = supabase.table("systems").select("compatibility_graph").eq("id", system_id).execute()
    
    if not system_response.data:
        raise HTTPException(status_code=404, detail=f"System with ID {system_id} not found")
    
    current_graph = system_response.data[0].get("compatibility_graph", {"edges": []})
    
    # Ensure current_graph has an edges array
    if not current_graph or "edges" not in current_graph:
        current_graph = {"edges": []}
    
    # Track existing edges to avoid duplicates
    existing_edge_keys = set()
    for edge in current_graph.get("edges", []):
        part_ids = edge.get("part_ids", [])
        if len(part_ids) >= 2:
            # Sort part IDs for consistent representation
            part_ids = sorted(part_ids)
            edge_key = f"{part_ids[0]}_{part_ids[1]}"
            existing_edge_keys.add(edge_key)
    
    # Add new edges if they don't already exist
    edges_to_add = []
    for edge in new_edges:
        part_ids = edge.get("part_ids", [])
        if len(part_ids) < 2:
            continue  # Skip invalid edges
        
        # Ensure part_ids are sorted
        part_ids = sorted(part_ids)
        edge["part_ids"] = part_ids
        
        # Create edge key for duplicate checking
        edge_key = f"{part_ids[0]}_{part_ids[1]}"
        
        if edge_key not in existing_edge_keys:
            edges_to_add.append(edge)
            existing_edge_keys.add(edge_key)
    
    # Update compatibility graph with new edges
    if edges_to_add:
        current_graph["edges"].extend(edges_to_add)
        
        # Update the system record
        update_response = supabase.table("systems").update({"compatibility_graph": current_graph}).eq("id", system_id).execute()
        
        if not update_response.data:
            raise HTTPException(status_code=500, detail=f"Failed to update compatibility graph for system {system_id}")

# --- API Endpoints ---
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Part Compatibility API", "status": "running"}

@app.post("/api/compatibility-check", response_model=List[CompatibilityCheckResponse])
async def check_compatibility(request: CompatibilityCheckRequest):
    """
    Check compatibility between multiple parts
    
    Args:
        request: Compatibility check request containing part IDs
        
    Returns:
        List of compatibility check results
    """
    part_ids = request.part_ids
    
    if not part_ids:
        raise HTTPException(status_code=400, detail="No part IDs provided")
    
    if len(part_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two part IDs are required for compatibility check")
    
    try:
        # 1. Get parts information
        parts = get_parts_info(part_ids)
        
        # 2. Get systems for updating the compatibility graph later
        # We still need subsystems and systems for the LLM format and to update compatibility graphs
        subsystems, systems = get_subsystems_and_systems_for_parts(part_ids)
        
        # 3. Format LLM input with specified field order
        formatted_parts = format_llm_input(parts, subsystems, systems)
        
        # 4. Perform compatibility check using Gemini LLM
        compatibility_results = check_compatibility_with_llm(formatted_parts)
        
        # 5. Extract compatibility edges (without subsystem info)
        edges = extract_compatibility_edges(compatibility_results)
        
        # 6. Update compatibility graphs in all related systems
        for system in systems:
            update_system_compatibility_graph(system["id"], edges)
        
        # 7. Return compatibility results
        return compatibility_results
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing compatibility check: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))