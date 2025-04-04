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
            "name": part.get("part_name", ""),  # part_name 사용
            "product_name": part.get("product_name", ""),  # product_name 명시적으로 포함
            "specifications": part.get("specifications", {}),
            "description": part.get("part_description", ""),
            "dimensions": part.get("dimensions", {}),  # 치수 정보 추가
            "weight": part.get("weight", {})  # 무게 정보 추가
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
            "part_ids": [str(pid) for pid in subsystem.get("part_ids", [])]
        }
        subsystems_info.append(subsystem_info)
        
    # Format systems information
    systems_info = []
    for system in systems:
        # ID를 문자열로 변환하고 subsystem_ids도 모두 문자열로 변환
        system_info = {
            "id": str(system["id"]),
            "description": system.get("system_description", ""),
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
    # 시스템 인스트럭션 정의
    system_instruction = """**Goal:**

You must evaluate and verify the compatibility of mechanical parts with other parts in the subsystem using their detailed specifications, dimensions, and attributes.

**Return Format (JSON):**

```json
[
  {
    "part_id": "string",
    "available": "boolean",
    "incompatibility": "object"
  }
]
```

**Note:**

- Ensure that a part's `available` status is set to `false` **only when it has a direct incompatibility** with another part.
- You will receive a structured JSON input with the following hierarchy:
    - **parts**: An array of part objects with these fields:
        - **id:** A string that uniquely identifies the part.
        - **name:** The specific part name.
        - **product_name:** The exact, catalog-specific part model name.
        - **specifications:** Key-value pairs detailing every critical parameter (values as strings).
        - **description:** A one-sentence, clear description of the part.
        - **dimensions:** An object containing information about the part's dimensions.
        - **weight:** An object with the part's weight information.
    - **subsystems**: An array of subsystem objects with these fields:
        - **id:** A string that uniquely identifies the subsystem.
        - **name:** The name of the subsystem.
        - **description:** A description of the subsystem highlighting its features.
        - **part_ids:** An array of strings containing the IDs of parts in this subsystem.
    - **systems**: An array of system objects with these fields:
        - **id:** A string that uniquely identifies the system.
        - **description:** A one-sentence, clear description of the entire system.
        - **subsystem_ids:** An array of strings containing the IDs of subsystems in this system.
- In your output:
    - Return an array containing one object for each part in the input.
    - Each part object must have:
        - **part_id:** A string that uniquely identifies the part.
        - **available:** A boolean indicating whether the part is compatible with other parts in the system.
        - **incompatibility:**
            - If **available** is `true`, provide an empty object {}.
            - If **available** is `false`, provide an object where each key is an incompatible part's id (as string) and each value is a string explaining the reason for the incompatibility.
- **Criteria for Compatibility and Interchangeability Inspection Between Parts:**
    - **Physical Interface Compatibility**
        - *Dimensional Verification*: Confirm critical dimensions and tolerance matching.
        - *Mechanical Connections*: Ensure exact matching of threads, keyways, splines, flanges, etc.
        - *Mounting Structure*: Verify that mounting hole positions, diameters, and patterns match exactly.
    - **Functional Compatibility**
        - *Performance Parameter Verification*: Compare key performance indicators such as torque, output, flow rate, pressure, etc.
        - *Operating Range Compatibility*: Confirm compatibility of RPM, load range, pressure range, etc.
        - *System Responsiveness*: Verify that response time and acceleration/deceleration characteristics match.
        - *Operating Characteristics*: Evaluate the impact of vibration, noise, and heat generation on overall system performance.
    - **Electrical/Electronic Compatibility**
        - *Electrical Specifications*: Match voltage, current, impedance, and frequency requirements.
        - *Connector Compatibility*: Verify matching of pin layout, connector type, and size.
        - *Signal Interface*: Ensure communication protocols and signal levels are compatible.
        - *EMI/EMC Characteristics*: Evaluate electromagnetic interference generation and immunity.
    - **Material and Environmental Compatibility**
        - *Thermal Expansion Characteristics*: Predict issues caused by differences in thermal expansion coefficients between materials.
        - *Temperature Influence Zone*: Assess the thermal impact of heat-generating components on surrounding parts.
    - **Practical Verification Methods**
        - *Durability Testing*: Verify compatibility issues such as wear and fatigue during long-term use.
        - *Boundary Condition Testing*: Verify performance under extreme conditions including maximum/minimum loads, temperatures, speeds, etc."""

    # Format input data
    input_data = format_llm_input(parts_info, subsystems, systems)
    
    # 모델 초기화 (시스템 인스트럭션 포함)
    model = genai.GenerativeModel(
        model_name=GOOGLE_MODEL_NAME,
        system_instruction=system_instruction
    )
    
    # 사용자 프롬프트 구성 및 메시지 형식으로 전달
    user_prompt = f"Analyze the compatibility of the following parts data: {input_data}"
    messages = [{"role": "user", "parts": [user_prompt]}]

    try:
        # Get response from LLM using messages format
        response = model.generate_content(messages)
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

async def update_system_compatibility_graph(edges: List[Dict[str, Any]], system_id: Optional[str] = None) -> None:
    """
    Update compatibility graph in Supabase systems table
    
    Args:
        edges: List of compatibility graph edges to update
        system_id: Optional system ID to update (if None, will attempt to find the system)
    """
    try:
        # If system_id is not provided, try to find the relevant system
        if not system_id and len(edges) > 0:
            # Get all systems to find the one that contains the parts
            response = supabase.table("systems").select("*").execute()
            systems = response.data
            
            # Find systems that might contain our parts
            part_ids = set()
            for edge in edges:
                part_ids.update(edge["part_ids"])
            
            # Find subsystems containing these parts
            subsystems_response = supabase.table("subsystems").select("*").execute()
            subsystems = subsystems_response.data
            
            relevant_subsystem_ids = set()
            for subsystem in subsystems:
                subsystem_part_ids = set(str(pid) for pid in subsystem.get("part_ids", []))
                if subsystem_part_ids.intersection(part_ids):
                    relevant_subsystem_ids.add(str(subsystem["id"]))
            
            # Find the system containing these subsystems
            for system in systems:
                system_subsystem_ids = set(str(sid) for sid in system.get("subsystem_ids", []))
                if system_subsystem_ids.intersection(relevant_subsystem_ids):
                    system_id = str(system["id"])
                    print(f"Found relevant system with ID: {system_id}")
                    break
        
        if system_id:
            # Get current compatibility graph for this system
            response = supabase.table("systems").select("compatibility_graph").eq("id", system_id).execute()
            
            if response.data and len(response.data) > 0:
                current_graph = response.data[0].get("compatibility_graph", {"edges": []})
                
                # Make sure the structure is as expected
                if not isinstance(current_graph, dict):
                    current_graph = {"edges": []}
                if "edges" not in current_graph:
                    current_graph["edges"] = []
                
                # Convert existing edges to set of part ID pairs for easy comparison
                existing_pairs = set()
                existing_edges = current_graph["edges"]
                for edge in existing_edges:
                    part_ids = [str(pid) for pid in edge.get("part_ids", [])]
                    if part_ids:
                        existing_pairs.add(tuple(sorted(part_ids)))
                
                # Add new edges
                new_edges_added = False
                for edge in edges:
                    # Ensure part_ids are strings
                    part_ids = [str(pid) for pid in edge.get("part_ids", [])]
                    part_ids = sorted(part_ids)
                    
                    # Check if edge already exists
                    if tuple(part_ids) not in existing_pairs:
                        current_graph["edges"].append({
                            "part_ids": part_ids,
                            "reason": edge.get("reason", "")
                        })
                        existing_pairs.add(tuple(part_ids))
                        new_edges_added = True
                
                # Update the system's compatibility graph if we added new edges
                if new_edges_added:
                    supabase.table("systems").update({"compatibility_graph": current_graph}).eq("id", system_id).execute()
                    print(f"Updated compatibility graph for system {system_id}")
                else:
                    print("No new edges to add to compatibility graph")
            else:
                # Create new compatibility graph
                compatibility_graph = {
                    "edges": edges
                }
                supabase.table("systems").update({"compatibility_graph": compatibility_graph}).eq("id", system_id).execute()
                print(f"Created new compatibility graph for system {system_id}")
        else:
            print("Warning: Could not determine which system to update compatibility graph for")
            
    except Exception as e:
        print(f"Error in update_system_compatibility_graph: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't raise an exception here, just log the error
        # This allows the API to continue returning results even if 
        # the compatibility graph can't be updated

@app.post("/api/compatibility-check")
async def compatibility_check(request: CompatibilityRequest):
    """
    Check compatibility between parts
    """
    try:
        print(f"Received compatibility check request: {json.dumps(request.dict())}")
        
        # 입력받은 part_ids가 모두 문자열인지 확인하고 변환
        request_part_ids = [str(pid) for pid in request.part_ids]
        print(f"Request part_ids (as strings): {request_part_ids}")
        
        # Get parts info
        parts_info = await get_parts_info(request_part_ids)
        if not parts_info:
            print("No parts found")
            raise HTTPException(status_code=404, detail="No parts found with the given IDs")
        print(f"Found {len(parts_info)} parts")
        
        # Get subsystems for parts
        subsystems = await get_subsystems_for_parts(request_part_ids)
        print(f"Found {len(subsystems)} subsystems")
        
        # ID가 문자열인지 확인
        subsystem_ids = [s["id"] for s in subsystems]
        print(f"Subsystem IDs for systems query: {subsystem_ids}")
        for i, sid in enumerate(subsystem_ids):
            if not isinstance(sid, str):
                print(f"Converting subsystem ID at index {i} from {type(sid)} to string")
                subsystem_ids[i] = str(sid)
        
        # Get systems for subsystems
        systems = await get_systems_for_subsystems(subsystem_ids)
        print(f"Found {len(systems)} systems")
        
        # Track system_id if found
        system_id = None
        if systems and len(systems) > 0:
            system_id = systems[0]["id"]
            print(f"Using system ID: {system_id} for compatibility graph")
        
        # Check compatibility with LLM
        llm_response = await check_compatibility_with_llm(parts_info, subsystems, systems)
        print(f"Received LLM response with {len(llm_response)} items")
        
        # Extract compatibility edges
        edges = extract_compatibility_edges(llm_response)
        print(f"Extracted {len(edges)} compatibility edges")
        
        # Update compatibility graph
        try:
            # Pass the system_id if we found one
            await update_system_compatibility_graph(edges, system_id)
            print("Updated compatibility graph")
        except Exception as e:
            # If updating the graph fails, just log it but continue
            print(f"Warning: Failed to update compatibility graph: {str(e)}")
            print("Continuing to return compatibility results...")
        
        return {
            "parts": parts_info,
            "subsystems": subsystems,
            "systems": systems,
            "compatibility_edges": edges,
            "compatibility_results": llm_response  # Also include the direct LLM response
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
        # Get parts info
        parts = await get_parts_info(request.part_ids)
        
        # Get subsystems for parts
        subsystems = await get_subsystems_for_parts(request.part_ids)
        
        # Get systems for subsystems
        systems = await get_systems_for_subsystems([s["id"] for s in subsystems])
        
        return {
            "parts": parts,
            "subsystems": subsystems,
            "systems": systems
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in debug endpoint: {str(e)}"
        )