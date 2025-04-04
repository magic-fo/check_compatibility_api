import json
import os
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai.types import Content, Part

# 환경 변수 로드
load_dotenv()

# Supabase 설정
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Gemini API 키
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Gemini 클라이언트 설정
genai.configure(api_key=gemini_api_key)
genai_client = genai.Client(api_key=gemini_api_key, http_options={'api_version':'v1alpha'})

# FastAPI 앱 생성
app = FastAPI()

# 요청 모델
class CompatibilityRequest(BaseModel):
    part_ids: List[Union[str, int]]

async def get_parts_info(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get part information from Supabase
    
    Args:
        part_ids: List of part IDs to get information for
        
    Returns:
        List of parts with information
    """
    try:
        parts = []
        
        for part_id in part_ids:
            # 문자열로 변환된 ID로 쿼리
            response = supabase.table("parts").select("*").eq("id", part_id).execute()
            
            if response.data and len(response.data) > 0:
                part = response.data[0]
                parts.append(part)
        
        if not parts:
            print(f"[ERROR] No parts found with IDs: {part_ids}")
            return []
            
        return parts
        
    except Exception as e:
        print(f"[ERROR] Error getting parts info: {str(e)}")
        return []

async def get_subsystems_for_parts(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get subsystems containing the specified parts
    
    Args:
        part_ids: List of part IDs
        
    Returns:
        List of subsystems containing the parts
    """
    try:
        subsystems = []
        subsystem_ids = set()  # 중복 제거를 위한 set
        
        for part_id in part_ids:
            try:
                # 해당 part_id를 포함하는 모든 subsystem 찾기
                response = supabase.table("subsystems").select("*").execute()
                
                for subsystem in response.data:
                    # 문자열로 변환하여 비교
                    subsystem_part_ids = [str(p_id) for p_id in subsystem.get("part_ids", [])]
                    
                    if str(part_id) in subsystem_part_ids and subsystem["id"] not in subsystem_ids:
                        subsystem_ids.add(subsystem["id"])
                        subsystems.append(subsystem)
            except Exception as e:
                print(f"[ERROR] Error getting subsystems for part ID {part_id}: {str(e)}")
        
        return subsystems
        
    except Exception as e:
        print(f"[ERROR] Error getting subsystems: {str(e)}")
        return []

async def get_systems_for_subsystems(subsystem_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get systems containing the specified subsystems
    
    Args:
        subsystem_ids: List of subsystem IDs
        
    Returns:
        List of systems containing the subsystems
    """
    try:
        systems = []
        system_ids = set()  # 중복 제거를 위한 set
        
        for subsystem_id in subsystem_ids:
            try:
                # 해당 subsystem_id를 포함하는 모든 system 찾기
                response = supabase.table("systems").select("*").execute()
                
                for system in response.data:
                    # 문자열로 변환하여 비교
                    system_subsystem_ids = [str(s_id) for s_id in system.get("subsystem_ids", [])]
                    
                    if str(subsystem_id) in system_subsystem_ids and system["id"] not in system_ids:
                        system_ids.add(system["id"])
                        systems.append(system)
            except Exception as e:
                print(f"[ERROR] Error getting systems for subsystem ID {subsystem_id}: {str(e)}")
        
        return systems
        
    except Exception as e:
        print(f"[ERROR] Error getting systems: {str(e)}")
        return []

def format_llm_input(parts: List[Dict[str, Any]], subsystems: List[Dict[str, Any]], systems: List[Dict[str, Any]]) -> str:
    """
    Format parts, subsystems, and systems information for LLM input
    
    Args:
        parts: List of parts
        subsystems: List of subsystems
        systems: List of systems
        
    Returns:
        Formatted input for the LLM
    """
    # 부품 정보 포맷팅
    parts_info = []
    for part in parts:
        part_info = {
            "id": str(part["id"]),
            "part_name": part.get("part_name", ""),
            "product_name": part.get("product_name", ""),
            "part_description": part.get("part_description", ""),
            "specifications": part.get("specifications", {})
        }
        parts_info.append(part_info)
    
    # 서브시스템 정보 포맷팅
    subsystems_info = []
    for subsystem in subsystems:
        subsystem_info = {
            "id": str(subsystem["id"]),
            "subsystem_name": subsystem.get("subsystem_name", ""),
            "subsystem_description": subsystem.get("subsystem_description", ""),
            "technical_engineering_specifications": subsystem.get("technical_engineering_specifications", [])
        }
        subsystems_info.append(subsystem_info)
    
    # 시스템 정보 포맷팅
    systems_info = []
    for system in systems:
        system_info = {
            "id": str(system["id"]),
            "system_name": system.get("system_name", ""),
            "system_description": system.get("system_description", ""),
            "system_specifications": system.get("system_specifications", {})
        }
        systems_info.append(system_info)
    
    # 전체 입력 구성
    formatted_input = json.dumps({
        "parts": parts_info,
        "subsystems": subsystems_info,
        "systems": systems_info
    })
    
    return formatted_input

async def check_compatibility_with_llm(parts_info: List[Dict[str, Any]], subsystems: List[Dict[str, Any]], systems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check compatibility between parts using Gemini 2.0 Flash Thinking
    
    Args:
        parts_info: List of parts to check compatibility for
        subsystems: List of subsystems containing the parts
        systems: List of systems containing the subsystems
        
    Returns:
        LLM response with compatibility results
    """
    try:
        # Format input data for LLM
        input_data = format_llm_input(parts_info, subsystems, systems)
        
        # 시스템 인스트럭션 설정
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
            - If **available** is `true`, provide an empty object `{}`.
            - If **available** is `false`, provide an object where each key is an incompatible part's id (as string) and each value is a string explaining the reason for the incompatibility.
- **Criteria for Compatibility and Interchangeability Inspection Between Parts:**
    - **Software & Firmware Compatibility:**
        
        Ensure that software and firmware versions, communication protocols, update configurations, and system settings are fully aligned with subsystem requirements. Implement dynamic update handling and exception logging to manage changes and maintain continuous compatibility.
        
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
        
        # 쿼리 구성
        query = f"""Check the compatibility between the parts provided.
{input_data}"""

        # 구조화된 입력 생성
        contents = Content(
            role="user",
            parts=[Part.from_text(text=query)]
        )
        
        # 새로운 SDK를 사용하여 Gemini 모델 호출
        response = genai_client.models.generate_content(
            model='gemini-2.0-flash-thinking-exp',
            contents=contents,
            system_instructions=system_instruction,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )
        
        try:
            # 응답에서 JSON 추출
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    text_response = candidate.content.parts[0].text
                    results = json.loads(text_response)
                    return results
                else:
                    print("[ERROR] Candidate content structure unexpected")
                    return []
            elif hasattr(response, 'text'):
                # 대체 방법으로 text 속성 사용
                text_response = response.text
                results = json.loads(text_response)
                return results
            else:
                print("[ERROR] Unexpected response format from Gemini API")
                return []
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[ERROR] Failed to parse LLM response as JSON: {str(e)}")
            return []
                
    except Exception as e:
        print(f"[ERROR] Error in LLM compatibility check: {str(e)}")
        return []

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
    
    print(f"[INFO] Starting extraction of compatibility edges")
    
    for part_result in llm_response:
        part_id = part_result.get("part_id")
        
        if not part_id or part_result.get("available", True):
            # Skip compatible parts
            continue
        
        incompatibility = part_result.get("incompatibility", {})
        if not incompatibility:
            continue
        
        for incompatible_part_id_str, reason in incompatibility.items():
            try:
                # 문자열 ID 사용
                incompatible_part_id = incompatible_part_id_str
                
                # LLM 응답으로부터 받은 part_id와 incompatible_part_id는 모두 문자열이어야 함
                part_id_str = str(part_id)
                incompatible_part_id_str = str(incompatible_part_id)
                
                # Sort part IDs to ensure consistent edge representation
                part_ids = sorted([part_id_str, incompatible_part_id_str])
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
                print(f"[INFO] Found incompatibility between parts {part_ids[0]} and {part_ids[1]}")
                
            except Exception as e:
                # Skip invalid part IDs
                print(f"[ERROR] Failed to process incompatibility: {str(e)}")
                continue
    
    print(f"[INFO] Completed extraction of compatibility edges: {len(edges)} found")
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
            print(f"[INFO] No system_id provided, attempting to find relevant system")
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
                    print(f"[INFO] Found relevant system with ID: {system_id}")
                    break
        
        if system_id:
            print(f"[INFO] Updating compatibility graph for system {system_id}")
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
                new_edges_count = 0
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
                        new_edges_count += 1
                
                # Update the system's compatibility graph if we added new edges
                if new_edges_added:
                    supabase.table("systems").update({"compatibility_graph": current_graph}).eq("id", system_id).execute()
                    print(f"[SUCCESS] Updated compatibility graph for system {system_id}. Added {new_edges_count} new edges.")
                else:
                    print(f"[INFO] No new edges to add to compatibility graph for system {system_id}")
            else:
                # Create new compatibility graph
                compatibility_graph = {
                    "edges": edges
                }
                supabase.table("systems").update({"compatibility_graph": compatibility_graph}).eq("id", system_id).execute()
                print(f"[SUCCESS] Created new compatibility graph for system {system_id} with {len(edges)} edges")
        else:
            print("[WARNING] Could not determine which system to update compatibility graph for")
            
    except Exception as e:
        print(f"[ERROR] Failed to update compatibility graph: {str(e)}")
        # Don't raise an exception here, just log the error
        # This allows the API to continue returning results even if 
        # the compatibility graph can't be updated

@app.post("/api/compatibility-check")
async def compatibility_check(request: CompatibilityRequest):
    """
    Check compatibility between parts and update compatibility graph in the database
    """
    try:
        print(f"[START] Compatibility check for part_ids: {request.part_ids}")
        
        # 입력받은 part_ids가 모두 문자열인지 확인하고 변환
        request_part_ids = [str(pid) for pid in request.part_ids]
        
        # Get parts info
        parts_info = await get_parts_info(request_part_ids)
        if not parts_info:
            print("[ERROR] No parts found with the given IDs")
            raise HTTPException(status_code=404, detail="No parts found with the given IDs")
        print(f"[INFO] Found {len(parts_info)} parts")
        
        # Get subsystems for parts
        subsystems = await get_subsystems_for_parts(request_part_ids)
        print(f"[INFO] Found {len(subsystems)} subsystems")
        
        # ID가 문자열인지 확인
        subsystem_ids = [s["id"] for s in subsystems]
        for i, sid in enumerate(subsystem_ids):
            if not isinstance(sid, str):
                subsystem_ids[i] = str(sid)
        
        # Get systems for subsystems
        systems = await get_systems_for_subsystems(subsystem_ids)
        print(f"[INFO] Found {len(systems)} systems")
        
        # Track system_id if found
        system_id = None
        if systems and len(systems) > 0:
            system_id = systems[0]["id"]
            print(f"[INFO] Using system ID: {system_id} for compatibility graph")
        else:
            print("[WARNING] No system found for these parts. Compatibility graph will not be updated.")
            return {
                "status": "warning",
                "message": "No system found for these parts. Compatibility check completed but graph not updated."
            }
        
        # Check compatibility with LLM
        try:
            llm_response = await check_compatibility_with_llm(parts_info, subsystems, systems)
            print(f"[INFO] LLM compatibility check completed successfully")
            
            # Extract compatibility edges
            edges = extract_compatibility_edges(llm_response)
            print(f"[INFO] Extracted {len(edges)} compatibility edges")
            
            # Update compatibility graph
            await update_system_compatibility_graph(edges, system_id)
            print(f"[SUCCESS] Compatibility graph updated for system {system_id}")
            
            return {
                "status": "success",
                "message": "Compatibility graph updated successfully",
                "system_id": system_id
            }
            
        except Exception as e:
            print(f"[ERROR] Error in compatibility process: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing compatibility check: {str(e)}",
                "system_id": system_id
            }
        
    except Exception as e:
        print(f"[ERROR] Error in compatibility check: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing compatibility check: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)