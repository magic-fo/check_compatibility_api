import json
import os
import sys
from typing import List, Dict, Any, Optional, Union
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# 필요한 최소 로깅만 유지
print(f"Python version: {sys.version}")
print(f"Starting API initialization")

# 환경 변수 로드
load_dotenv()

# Flask 앱 생성 - 가장 먼저 초기화
app = Flask(__name__)

# Supabase 설정
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

print(f"Supabase URL available: {bool(supabase_url)}")
print(f"Supabase Key available: {bool(supabase_key)}")

try:
    from supabase import create_client, Client
    if supabase_url and supabase_key:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("Supabase client initialized successfully")
    else:
        supabase = None
        print("[WARNING] Supabase credentials not found, some features will be limited")
except Exception as e:
    print(f"[ERROR] Failed to initialize Supabase client: {str(e)}")
    supabase = None

# Gemini API 키
gemini_api_key = os.environ.get("GOOGLE_API_KEY")
gemini_model_name = os.environ.get("GOOGLE_MODEL_NAME", "gemini-1.5-flash-thinking")
print(f"Gemini API Key available: {bool(gemini_api_key)}")
print(f"Using model: {gemini_model_name}")

# Gemini 클라이언트 설정
genai = None
try:
    if gemini_api_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        print("Gemini client initialized successfully")
    else:
        print("[WARNING] Gemini API key not found, LLM features will be unavailable")
except Exception as e:
    print(f"[ERROR] Failed to initialize Gemini client: {str(e)}")
    genai = None

def get_parts_info(part_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get part information from Supabase
    
    Args:
        part_ids: List of part IDs to get information for
        
    Returns:
        List of parts with information
    """
    if not supabase:
        print("[ERROR] Supabase client not initialized")
        return []
        
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

def get_subsystems_for_parts(part_ids: List[str]) -> List[Dict[str, Any]]:
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

def get_systems_for_subsystems(subsystem_ids: List[str]) -> List[Dict[str, Any]]:
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
            "name": part.get("part_name", ""),
            "product_name": part.get("product_name", ""),
            "specifications": part.get("specifications", {}),
            "description": part.get("part_description", ""),
            "dimensions": part.get("dimensions", {}),
            "weight": part.get("weight", {})
        }
        parts_info.append(part_info)
    
    # 서브시스템 정보 포맷팅
    subsystems_info = []
    for subsystem in subsystems:
        subsystem_info = {
            "id": str(subsystem["id"]),
            "name": subsystem.get("subsystem_name", ""),
            "description": subsystem.get("subsystem_description", ""),
            "part_ids": [str(p_id) for p_id in subsystem.get("part_ids", [])]
        }
        subsystems_info.append(subsystem_info)
    
    # 시스템 정보 포맷팅
    systems_info = []
    for system in systems:
        system_info = {
            "id": str(system["id"]),
            "description": system.get("system_description", ""),
            "subsystem_ids": [str(s_id) for s_id in system.get("subsystem_ids", [])]
        }
        systems_info.append(system_info)
    
    # 전체 입력 구성
    formatted_input = json.dumps({
        "parts": parts_info,
        "subsystems": subsystems_info,
        "systems": systems_info
    })
    
    return formatted_input

def check_compatibility_with_llm(parts_info: List[Dict[str, Any]], subsystems: List[Dict[str, Any]], systems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Check compatibility between parts using Gemini model
    
    Args:
        parts_info: List of parts to check compatibility for
        subsystems: List of subsystems containing the parts
        systems: List of systems containing the subsystems
        
    Returns:
        LLM response with compatibility results
    """
    if not genai:
        print("[ERROR] Gemini client not initialized")
        return []
        
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
        
        # 간소화된 쿼리 생성
        query = f"""Check the compatibility between the parts provided.
{input_data}"""
        
        # Gemini API 호출
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }
        
        # 모델 설정 - 환경변수에서 가져온 모델 이름 사용
        model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config=generation_config,
        )
        
        # 시스템 인스트럭션 및 콘텐츠 설정
        chat = model.start_chat(system_instruction=system_instruction)
        response = chat.send_message(query)
        
        # 응답 파싱
        try:
            # 텍스트에서 JSON 부분 추출
            result_text = response.text
            if "```json" in result_text and "```" in result_text:
                json_text = result_text.split("```json")[1].split("```")[0].strip()
            else:
                json_text = result_text
                
            # JSON 파싱
            compatibility_result = json.loads(json_text)
            return compatibility_result
        except Exception as e:
            print(f"[ERROR] Failed to parse Gemini response: {str(e)}")
            print(f"Raw response: {response.text}")
            return []
    
    except Exception as e:
        print(f"[ERROR] Error checking compatibility with LLM: {str(e)}")
        return []

def extract_compatibility_edges(llm_response: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

def update_system_compatibility_graph(edges: List[Dict[str, Any]], system_id: Optional[str] = None) -> None:
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
            
            # 새 그래프 구성 (기존 그래프와 병합하지 않고 완전히 대체)
            new_compatibility_graph = {
                "edges": edges  # 새로운 엣지만 사용
            }
            
            # 데이터베이스 업데이트 - 기존 그래프 덮어쓰기
            supabase.table("systems").update({
                "compatibility_graph": new_compatibility_graph
            }).eq("id", system_id).execute()
            
            print(f"[SUCCESS] Replaced compatibility graph for system {system_id} with {len(edges)} edges")
        else:
            print("[WARNING] Could not determine which system to update compatibility graph for")
            
    except Exception as e:
        print(f"[ERROR] Failed to update compatibility graph: {str(e)}")
        # Don't raise an exception here, just log the error

@app.route('/api/compatibility-check', methods=['POST'])
def compatibility_check():
    """
    Check compatibility between parts
    """
    if not supabase:
        return jsonify({"error": "Database connection not available"}), 500
        
    if not genai:
        return jsonify({"error": "LLM service not available"}), 500
    
    try:
        # 요청 데이터 가져오기
        data = request.get_json()
        if not data or "part_ids" not in data:
            return jsonify({"error": "Invalid request. 'part_ids' field is required"}), 400
            
        # 문자열로 변환
        part_ids = [str(id) for id in data["part_ids"]]
        
        if len(part_ids) < 2:
            return jsonify({"error": "At least 2 parts are required for compatibility check"}), 400
        
        # 부품 정보 조회
        parts = get_parts_info(part_ids)
        
        if not parts:
            return jsonify({"error": "No parts found with the provided IDs"}), 404
        
        # 서브시스템 정보 조회
        subsystems = get_subsystems_for_parts(part_ids)
        
        # 시스템 정보 조회
        system_ids = []
        if subsystems:
            subsystem_ids = [str(subsystem["id"]) for subsystem in subsystems]
            systems = get_systems_for_subsystems(subsystem_ids)
            if systems:
                system_ids = [str(system["id"]) for system in systems]
        else:
            systems = []
        
        # LLM으로 호환성 체크
        compatibility_results = check_compatibility_with_llm(parts, subsystems, systems)
        
        if not compatibility_results:
            return jsonify({"error": "Failed to check compatibility"}), 500
        
        # 엣지 추출
        edges = extract_compatibility_edges(compatibility_results)
        
        # 결과를 데이터베이스에 저장
        if systems and len(systems) > 0:
            # 호환성 그래프를 첫 번째 시스템에 연결
            system_id = systems[0]["id"]
            update_system_compatibility_graph(edges, system_id)
        
        return jsonify({
            "compatibility_results": compatibility_results,
            "part_ids": part_ids,
            "systems": [system["id"] for system in systems] if systems else []
        })
        
    except Exception as e:
        print(f"[ERROR] Error in compatibility check: {str(e)}")
        return jsonify({"error": f"Error checking compatibility: {str(e)}"}), 500

@app.route('/')
def root():
    """루트 엔드포인트"""
    available_modules = []
    try:
        import sys
        available_modules.append("sys")
    except ImportError:
        pass
    
    try:
        import google
        available_modules.append("google")
        
        # Google 모듈 세부 정보 확인
        google_modules = dir(google)
    except ImportError:
        google_modules = "not available"
    
    # 환경 정보 수집
    status = {
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "/api/compatibility-check": "POST - Check compatibility between parts",
            "/env-check": "GET - Check environment variables"
        },
        "environments": {
            "supabase": "available" if supabase else "unavailable",
            "gemini": "available" if genai else "unavailable"
        },
        "debug": {
            "python_version": sys.version,
            "available_modules": available_modules,
            "google_modules": google_modules if "google" in available_modules else None,
            "env_vars": {k: bool(v) for k, v in os.environ.items() if not k.startswith("AWS_")},
        }
    }
    return jsonify(status)

@app.route('/env-check')
def env_check():
    """환경 변수 확인 엔드포인트"""
    env_vars = {
        "SUPABASE_URL": bool(os.environ.get("SUPABASE_URL")),
        "SUPABASE_KEY": bool(os.environ.get("SUPABASE_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "PYTHON_VERSION": sys.version
    }
    return jsonify(env_vars)

# Vercel 서버리스 함수 핸들러
def handler(event, context):
    """Vercel 서버리스 함수 핸들러"""
    path = event.get('path', '/')
    http_method = event.get('httpMethod', 'GET')
    
    # Flask 앱 컨텍스트 설정
    with app.test_request_context(
        path=path,
        method=http_method,
        headers=event.get('headers', {}),
        query_string=event.get('queryStringParameters', {}),
        data=event.get('body', '')
    ):
        # Flask에서 요청 처리
        try:
            response = app.full_dispatch_request()
            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data(as_text=True)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)