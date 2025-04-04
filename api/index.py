import json
import os
import sys
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from mangum import Mangum

# 필요한 최소 로깅만 유지
print(f"Python version: {sys.version}")
print(f"Starting API initialization")

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성 - 가장 먼저 초기화
app = FastAPI()

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
gemini_api_key = os.environ.get("GEMINI_API_KEY")
print(f"Gemini API Key available: {bool(gemini_api_key)}")

# Gemini 클라이언트 설정 - 변경된 임포트 방식
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
    "part_ids": ["part_id_1", "part_id_2"],
    "compatible": true/false,
    "reason": "Detailed explanation of compatibility or incompatibility",
    "confidence": 0.9 // 0.0-1.0 scale
  },
  // Additional entries for other part pairs
]
```

**Decision Criteria:**
1. Physical dimensions matching
2. Interface compatibility
3. Electrical compatibility (if applicable)
4. Signal/data compatibility (if applicable)
5. Performance matching
6. Environmental condition compatibility
7. Material compatibility

**Guidelines:**
- Only identify direct compatibility between specific parts
- Be conservative: if compatibility is uncertain, mark as incompatible
- Analyze all possible part pairs
- Consider both technical specifications and functional requirements
- For each compatibility evaluation, provide detailed technical reasons"""
        
        # 간소화된 쿼리 생성 (사용자 요청에 따라)
        query = f"""Check the compatibility between the parts provided.
{input_data}"""
        
        # 최신 Gemini API 호출 방식으로 업데이트
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 2048,
        }
        
        # 모델 설정
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
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
        # This allows the API to continue returning results even if 
        # the compatibility graph can't be updated

@app.post("/api/compatibility-check")
async def compatibility_check(request: CompatibilityRequest):
    """
    Check compatibility between parts
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available")
        
    if not genai:
        raise HTTPException(status_code=500, detail="LLM service not available")
    
    try:
        # 문자열로 변환
        part_ids = [str(id) for id in request.part_ids]
        
        if len(part_ids) < 2:
            return {"error": "At least 2 parts are required for compatibility check"}
        
        # 부품 정보 조회
        parts = await get_parts_info(part_ids)
        
        if not parts:
            raise HTTPException(status_code=404, detail="No parts found with the provided IDs")
        
        # 서브시스템 정보 조회
        subsystems = await get_subsystems_for_parts(part_ids)
        
        # 시스템 정보 조회
        system_ids = []
        if subsystems:
            subsystem_ids = [str(subsystem["id"]) for subsystem in subsystems]
            systems = await get_systems_for_subsystems(subsystem_ids)
            if systems:
                system_ids = [str(system["id"]) for system in systems]
        else:
            systems = []
        
        # LLM으로 호환성 체크
        compatibility_results = await check_compatibility_with_llm(parts, subsystems, systems)
        
        if not compatibility_results:
            return {"error": "Failed to check compatibility"}
        
        # 엣지 추출
        edges = extract_compatibility_edges(compatibility_results)
        
        # 결과를 데이터베이스에 저장
        if systems and len(systems) > 0:
            # 호환성 그래프를 첫 번째 시스템에 연결
            system_id = systems[0]["id"]
            await update_system_compatibility_graph(edges, system_id)
        
        return {
            "compatibility_results": compatibility_results,
            "part_ids": part_ids,
            "systems": [system["id"] for system in systems] if systems else []
        }
        
    except Exception as e:
        print(f"[ERROR] Error in compatibility check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking compatibility: {str(e)}")

@app.get("/")
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
    return status

@app.get("/env-check")
def env_check():
    """환경 변수 확인 엔드포인트"""
    # 비동기 제거
    env_vars = {
        "SUPABASE_URL": bool(os.environ.get("SUPABASE_URL")),
        "SUPABASE_KEY": bool(os.environ.get("SUPABASE_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "PYTHON_VERSION": sys.version
    }
    return env_vars

# Mangum 핸들러 설정 - 앱 라우트 정의 후에 초기화
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)