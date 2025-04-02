import os
import json
import traceback
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_MODEL_NAME = os.environ.get("GOOGLE_MODEL_NAME", "gemini-2.0-flash-thinking-exp")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GOOGLE_MODEL_NAME)

app = FastAPI()

# 시작 로그 추가
print("Starting FastAPI application...")
print(f"GOOGLE_MODEL_NAME: {GOOGLE_MODEL_NAME}")
print(f"GOOGLE_API_KEY set: {bool(GOOGLE_API_KEY)}")
print(f"SUPABASE_URL set: {bool(supabase_url)}")
print(f"SUPABASE_KEY set: {bool(supabase_key)}")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompatibilityRequest(BaseModel):
    part_ids: List[str]

async def get_parts_info(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get part information from Supabase
    
    Args:
        part_ids: List of part IDs to get information for
        
    Returns:
        List of part information dictionaries
    """
    try:
        # Convert all part_ids to strings
        part_ids = [str(pid) for pid in part_ids]
        print(f"Querying parts with IDs (as strings): {part_ids}")
        
        # Query all parts at once
        response = supabase.table("parts").select("*").in_("id", part_ids).execute()
        
        if response.data:
            # Convert IDs to strings
            parts = []
            for part in response.data:
                part["id"] = str(part["id"])
                parts.append(part)
                print(f"Found part: {json.dumps(part)}")
            print(f"Parts after conversion: {json.dumps(parts)}")
            return parts
        else:
            print(f"No parts found with IDs: {part_ids}")
            return []
            
    except Exception as e:
        print(f"Error getting parts info: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

async def get_subsystems_for_parts(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get subsystems that contain the specified parts
    
    Args:
        part_ids: List of part IDs to get subsystems for
        
    Returns:
        List of subsystem dictionaries
    """
    subsystems = []
    seen_subsystem_ids = set()
    
    # Ensure input part_ids are strings
    part_ids = [str(pid) for pid in part_ids]
    print(f"Querying subsystems for part IDs (as strings): {part_ids}")

    for part_id in part_ids:
        try:
            # part_id를 문자열로 사용하여 쿼리
            response = supabase.table("subsystems").select("*").contains("part_ids", [part_id]).execute()
            
            for subsystem in response.data:
                subsystem_id_str = str(subsystem["id"]) # ID를 문자열로 변환
                if subsystem_id_str not in seen_subsystem_ids:
                    # 모든 관련 ID를 문자열로 변환하여 저장
                    subsystem["id"] = subsystem_id_str
                    subsystem["part_ids"] = [str(pid) for pid in subsystem.get("part_ids", [])]
                    subsystems.append(subsystem)
                    seen_subsystem_ids.add(subsystem_id_str)
                    print(f"Found subsystem (IDs as strings): {json.dumps(subsystem)}")

        except Exception as e:
            print(f"Error getting subsystems for part ID {part_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    print(f"Subsystems after conversion: {json.dumps(subsystems)}")
    return subsystems

async def get_systems_for_subsystems(subsystem_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get systems that contain the specified subsystems
    
    Args:
        subsystem_ids: List of subsystem IDs to get systems for
        
    Returns:
        List of system dictionaries
    """
    systems = []
    seen_system_ids = set()

    # Ensure input subsystem_ids are strings
    subsystem_ids = [str(sid) for sid in subsystem_ids]
    print(f"Querying systems for subsystem IDs (as strings): {subsystem_ids}")
    
    for subsystem_id in subsystem_ids:
        try:
            # subsystem_id를 문자열로 사용하여 쿼리
            response = supabase.table("systems").select("*").contains("subsystem_ids", [subsystem_id]).execute()
            
            for system in response.data:
                system_id_str = str(system["id"]) # ID를 문자열로 변환
                if system_id_str not in seen_system_ids:
                    # 모든 관련 ID를 문자열로 변환하여 저장
                    system["id"] = system_id_str
                    system["subsystem_ids"] = [str(sid) for sid in system.get("subsystem_ids", [])]
                    systems.append(system)
                    seen_system_ids.add(system_id_str)
                    print(f"Found system (IDs as strings): {json.dumps(system)}")

        except Exception as e:
            print(f"Error getting systems for subsystem ID {subsystem_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    print(f"Systems after conversion: {json.dumps(systems)}")
    return systems

def format_llm_input(parts: List[Dict[str, Any]], subsystems: List[Dict[str, Any]], systems: List[Dict[str, Any]]) -> str:
    """
    Format input for LLM compatibility check
    
    Args:
        parts: List of part information
        subsystems: List of subsystems containing the parts
        systems: List of systems containing the subsystems
        
    Returns:
        Formatted input string for LLM
    """
    # Format parts information
    parts_info = []
    for part in parts:
        part_info = {
            "id": str(part["id"]),  # ID를 문자열로 변환
            "name": part.get("product_name", part.get("part_name", "")),  # product_name이나 part_name 사용
            "category": part.get("category", ""),
            "specifications": part.get("specifications", {}),
            "description": part.get("part_description", "")
        }
        parts_info.append(part_info)
        
    # Format subsystems information
    subsystems_info = []
    for subsystem in subsystems:
        # ID를 문자열로 변환하고 part_ids도 모두 문자열로 변환
        subsystem_info = {
            "id": str(subsystem["id"]),
            "name": subsystem.get("subsystem_name", ""),  # subsystem_name 사용
            "description": subsystem.get("subsystem_description", ""),
            "requirements": subsystem.get("requirements", {}),
            "part_ids": [str(pid) for pid in subsystem.get("part_ids", [])]
        }
        subsystems_info.append(subsystem_info)
        
    # Format systems information
    systems_info = []
    for system in systems:
        # ID를 문자열로 변환하고 subsystem_ids도 모두 문자열로 변환
        system_info = {
            "id": str(system["id"]),
            "name": system.get("system_name", ""),  # system_name 사용
            "description": system.get("system_description", ""),
            "requirements": system.get("requirements", {}),
            "subsystem_ids": [str(sid) for sid in system.get("subsystem_ids", [])]
        }
        systems_info.append(system_info)
        
    # Combine all information
    input_data = {
        "parts": parts_info,
        "subsystems": subsystems_info,
        "systems": systems_info
    }
    
    # 디버깅 출력 추가
    formatted_input = json.dumps(input_data, indent=2)
    print(f"Formatted LLM input (IDs as strings): {formatted_input}")
    
    return formatted_input

async def check_compatibility_with_llm(parts_info: List[Dict[str, Any]], subsystems: List[Dict[str, Any]], systems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check compatibility between parts using LLM
    
    Args:
        parts_info: List of part information
        subsystems: List of subsystems containing the parts
        systems: List of systems containing the subsystems
        
    Returns:
        List of compatibility results
    """
    # Format input data
    input_data = format_llm_input(parts_info, subsystems, systems)
    
    # Create prompt
    prompt = f"""You are a drone engineering expert. Please analyze the compatibility between the provided drone parts.
The input data includes parts information, subsystems they belong to, and systems those subsystems are part of.

Input data:
{input_data}

Please check if there are any compatibility issues between the parts, considering:
1. Physical compatibility (size, mounting, etc.)
2. Electrical compatibility (voltage, current, etc.)
3. Performance compatibility (power requirements, etc.)
4. Protocol compatibility (communication standards, etc.)

Return a JSON array where each object represents a part and its compatibility status. Each object should have:
- part_id (string): The ID of the part being analyzed
- available (boolean): Whether the part is available and valid
- incompatibility (object): A dictionary mapping incompatible part IDs (as strings) to reasons for incompatibility
- notes (string): Any additional notes about the part's compatibility

Example response format:
[
  {{
    "part_id": "123",
    "available": true,
    "incompatibility": {{}},
    "notes": "Compatible with all other parts"
  }},
  {{
    "part_id": "456",
    "available": true,
    "incompatibility": {{
      "789": "Voltage mismatch - requires 12V but only supports 5V"
    }},
    "notes": "Partially compatible"
  }}
]

Important: 
- Keep part_id as a string, do not convert to number
- Only include incompatibility entries for actual incompatibilities
- Provide specific technical reasons for any incompatibilities
- Consider both direct and indirect compatibility requirements
"""

    try:
        # Get response from LLM
        response = model.generate_content(prompt)
        print(f"Raw LLM response: {response.text}")
        
        # Parse response
        response_text = response.text
        # Remove code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        # Parse JSON
        results = json.loads(response_text.strip())
        print(f"Parsed JSON: {json.dumps(results, indent=2)}")
        
        # Ensure all part_ids are strings
        for result in results:
            if "part_id" in result:
                result["part_id"] = str(result["part_id"])
                if "incompatibility" in result:
                    result["incompatibility"] = {
                        str(k): v for k, v in result["incompatibility"].items()
                    }
        
        return results
        
    except Exception as e:
        print(f"Error in LLM compatibility check: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Error in LLM compatibility check: {str(e)}")

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
    
    print(f"EXTRACTING EDGES FROM: {json.dumps(llm_response[:2], indent=2)}")
    
    for part_result in llm_response:
        part_id = part_result.get("part_id")
        print(f"Processing part_id: {part_id}, type: {type(part_id)}")
        
        if not part_id or part_result.get("available", True):
            print(f"Skipping part {part_id} - available: {part_result.get('available', True)}")
            continue  # Skip compatible parts
        
        incompatibility = part_result.get("incompatibility", {})
        if not incompatibility:
            print(f"No incompatibilities for part {part_id}")
            continue
        
        print(f"Incompatibilities for {part_id}: {json.dumps(incompatibility)}")
        
        for incompatible_part_id_str, reason in incompatibility.items():
            try:
                # 문자열 ID 사용
                incompatible_part_id = incompatible_part_id_str
                
                # LLM 응답으로부터 받은 part_id와 incompatible_part_id는 모두 문자열이어야 함
                part_id_str = str(part_id)
                incompatible_part_id_str = str(incompatible_part_id)
                
                print(f"Creating edge for {part_id_str} <-> {incompatible_part_id_str}")
                
                # Sort part IDs to ensure consistent edge representation
                part_ids = sorted([part_id_str, incompatible_part_id_str])
                pair_key = f"{part_ids[0]}_{part_ids[1]}"
                
                # Skip if this pair has already been processed
                if pair_key in processed_pairs:
                    print(f"Skipping duplicate pair: {pair_key}")
                    continue
                
                processed_pairs.add(pair_key)
                
                # Create edge
                edge = {
                    "part_ids": part_ids,
                    "reason": reason
                }
                
                edges.append(edge)
                print(f"Added edge: {json.dumps(edge)}")
                
            except Exception as e:
                # Skip invalid part IDs
                print(f"Error processing part ID: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"EXTRACTED EDGES: {json.dumps(edges)}")
    return edges

async def update_system_compatibility_graph(edges: List[Dict[str, Any]]) -> None:
    """
    Update compatibility graph in Supabase
    
    Args:
        edges: List of compatibility graph edges to update
    """
    try:
        # Get existing compatibility graph
        response = supabase.table("compatibility_graph").select("*").execute()
        existing_edges = response.data if response.data else []
        
        # Convert existing edges to set of part ID pairs for easy comparison
        existing_pairs = set()
        for edge in existing_edges:
            # part_ids를 문자열로 변환하여 저장
            part_ids = [str(pid) for pid in edge["part_ids"]]
            existing_pairs.add(tuple(sorted(part_ids)))
        
        # Add new edges
        for edge in edges:
            # edge의 part_ids를 문자열로 변환
            part_ids = [str(pid) for pid in edge["part_ids"]]
            part_ids = sorted(part_ids)
            
            # Check if edge already exists
            if tuple(part_ids) not in existing_pairs:
                # Insert new edge
                supabase.table("compatibility_graph").insert({
                    "part_ids": part_ids,
                    "reason": edge["reason"]
                }).execute()
                
    except Exception as e:
        print(f"Error updating compatibility graph: {str(e)}")
        raise Exception(f"Error updating compatibility graph: {str(e)}")

@app.post("/api/compatibility-check")
async def compatibility_check(request: CompatibilityRequest):
    """
    Check compatibility between parts
    """
    try:
        print(f"Received compatibility check request: {json.dumps(request.dict())}")
        
        # Get parts info
        parts_info = await get_parts_info(request.part_ids)
        print(f"Parts info: {json.dumps(parts_info)}")
        
        # Get subsystems for parts
        subsystems = await get_subsystems_for_parts(request.part_ids)
        print(f"Subsystems: {json.dumps(subsystems)}")
        
        # Get systems for subsystems
        systems = await get_systems_for_subsystems([s["id"] for s in subsystems])
        print(f"Systems: {json.dumps(systems)}")
        
        # Check compatibility with LLM
        llm_response = await check_compatibility_with_llm(parts_info, subsystems, systems)
        print(f"LLM response: {json.dumps(llm_response)}")
        
        # Extract compatibility edges
        edges = extract_compatibility_edges(llm_response)
        print(f"Extracted edges: {json.dumps(edges)}")
        
        # Update compatibility graph
        await update_system_compatibility_graph(edges)
        print("Updated compatibility graph")
        
        return {
            "parts": parts_info,
            "subsystems": subsystems,
            "systems": systems,
            "compatibility_edges": edges
        }
        
    except Exception as e:
        print(f"Error in compatibility check: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing compatibility check: {str(e)}"
        )

@app.post("/api/debug")
async def debug_endpoint(request: CompatibilityRequest):
    """
    Debug endpoint to check part information and system hierarchy
    """
    try:
        print(f"Received debug request: {json.dumps(request.dict())}")
        
        # 입력받은 part_ids가 모두 문자열인지 확인하고 변환
        request_part_ids = [str(pid) for pid in request.part_ids]
        print(f"Debug request part_ids (as strings): {request_part_ids}")
        
        # Get parts info
        parts = await get_parts_info(request_part_ids)
        if not parts:
            print("No parts found")
            raise HTTPException(status_code=404, detail="No parts found with the given IDs")
        print(f"Debug: Found {len(parts)} parts")
        
        # Get subsystems for parts
        subsystems = await get_subsystems_for_parts(request_part_ids)
        print(f"Debug: Found {len(subsystems)} subsystems")
        
        # ID가 문자열인지 확인
        subsystem_ids = [s["id"] for s in subsystems]
        print(f"Debug: Subsystem IDs for systems query: {subsystem_ids}")
        for i, sid in enumerate(subsystem_ids):
            if not isinstance(sid, str):
                print(f"Debug: Converting subsystem ID at index {i} from {type(sid)} to string")
                subsystem_ids[i] = str(sid)
        
        # Get systems for subsystems
        systems = await get_systems_for_subsystems(subsystem_ids)
        print(f"Debug: Found {len(systems)} systems")
        
        # 환경 변수 정보 추가
        env_info = {
            "SUPABASE_URL_SET": bool(supabase_url),
            "SUPABASE_KEY_SET": bool(supabase_key),
            "GOOGLE_API_KEY_SET": bool(GOOGLE_API_KEY),
            "GOOGLE_MODEL_NAME": GOOGLE_MODEL_NAME,
            "Current ENV": os.environ.get("ENVIRONMENT", "unknown")
        }
        
        return {
            "parts": parts,
            "subsystems": subsystems,
            "systems": systems,
            "env_info": env_info
        }
        
    except Exception as e:
        print(f"Error in debug endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error in debug endpoint: {str(e)}"
        )

@app.get("/api/env-check")
async def env_check():
    """
    Debug endpoint to check environment variables
    """
    try:
        print("Environment check endpoint called")
        # 모든 환경 변수 출력하지 않도록 필요한 것만 반환
        return {
            "SUPABASE_URL_SET": bool(supabase_url),
            "SUPABASE_KEY_SET": bool(supabase_key),
            "GOOGLE_API_KEY_SET": bool(GOOGLE_API_KEY),
            "GOOGLE_MODEL_NAME": GOOGLE_MODEL_NAME,
            "Current ENV": os.environ.get("ENVIRONMENT", "unknown")
        }
    except Exception as e:
        print(f"Error in env-check endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error in env-check endpoint: {str(e)}"
        )

# 루트 경로 추가
@app.get("/")
async def root():
    """
    Root endpoint for health check
    """
    return {"status": "ok", "message": "API is running"}

# API 테스트 경로 추가
@app.get("/api/test")
async def api_test():
    """
    Simple API test endpoint
    """
    return {"status": "ok", "message": "API is working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)