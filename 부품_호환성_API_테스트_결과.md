# 부품 호환성 API 테스트 결과 및 분석

## 개요

이 문서는 `https://check-compatibility-api.vercel.app/` 호스팅되는 부품 호환성 검사 API의 테스트 결과와 분석 내용을 담고 있습니다. API의 동작 방식, 입력/출력 구조, LLM 처리 과정 등을 자세히 설명합니다.

## API 구조 및 엔드포인트

API는 FastAPI를 사용하여 구현되었으며, 다음 엔드포인트를 제공합니다:

1. **`/api/compatibility-check`**: 부품 간 호환성 검사를 수행하는 메인 엔드포인트
2. **`/api/debug`**: 부품, 서브시스템, 시스템 정보를 조회하는 디버그 엔드포인트
3. **`/docs`**: FastAPI 자동 생성 API 문서 (Swagger UI)
4. **`/openapi.json`**: OpenAPI 스펙

## 요청 구조

모든 API 엔드포인트는 다음과 같은 JSON 형식의 요청을 받습니다:

```json
{
  "part_ids": ["2528", "2529", "2530"]
}
```

여기서 `part_ids`는 호환성을 검사할 부품의 ID 목록입니다. 모든 ID는 문자열 형태로 처리됩니다.

## 데이터 처리 흐름

### 1. 부품 및 시스템 정보 조회

요청 받은 부품 ID를 사용하여 Supabase 데이터베이스에서 다음 정보를 순차적으로 조회합니다:

1. **부품 정보**: 각 부품의 상세 정보 (부품명, 설명, 제품명, 사양 등)
2. **서브시스템 정보**: 해당 부품들을 포함하는 서브시스템 정보
3. **시스템 정보**: 해당 서브시스템들을 포함하는 시스템 정보

### 2. LLM 입력 데이터 형식화

조회한 정보를 LLM이 처리할 수 있는 구조화된 형식으로 변환합니다:

```json
{
  "parts": [
    {
      "id": "2528",
      "name": "Brushless DC Motor",
      "product_name": "T-Motor F80 Pro 2508",
      "specifications": {
        "kv": "2500",
        "max_power": "706W",
        "idle_current": "2.5A",
        "peak_current": "47.6A",
        "configuration": "12N14P",
        "rated_voltage": "3-6S",
        "shaft_diameter": "4mm",
        "internal_resistance": "22mΩ"
      },
      "description": "High-performance 2508 size brushless motor with 1000KV rating",
      "dimensions": {
        "width": {
          "unit": "mm",
          "value": 29
        },
        "height": {
          "unit": "mm",
          "value": 29
        },
        "length": {
          "unit": "mm",
          "value": 33.6
        }
      },
      "weight": {
        "unit": "g",
        "value": 39.7
      }
    },
    // ... 다른 부품 정보
  ],
  "subsystems": [
    {
      "id": "690",
      "name": "Propulsion System",
      "description": "Provides thrust for drone takeoff, flight, and landing.",
      "part_ids": ["2528", "2529", "2530", "2531", "2532"]
    }
  ],
  "systems": [
    {
      "id": "125",
      "description": "A quadcopter drone designed for general-purpose aerial operations...",
      "subsystem_ids": ["690", "693", "689", "688", "692", "691"]
    }
  ]
}
```

### 3. LLM 처리 과정

LLM(Google Gemini)에 시스템 인스트럭션과 함께 입력 데이터를 전달합니다:

**시스템 인스트럭션**:
```
**Goal:**
You must evaluate and verify the compatibility of mechanical parts with other parts in the subsystem using their detailed specifications, dimensions, and attributes.

**Return Format (JSON):**
[
  {
    "part_id": "string",
    "available": "boolean",
    "incompatibility": "object"
  }
]

**Note:**
- Ensure that a part's `available` status is set to `false` **only when it has a direct incompatibility** with another part.
...
```

**사용자 프롬프트**:
```
Analyze the compatibility of the following parts data: {input_data}
```

### 4. LLM 응답 형식

LLM은 다음과 같은 구조의 JSON 배열을 반환합니다:

```json
[
  {
    "part_id": "2528",
    "available": true,
    "incompatibility": {}
  },
  {
    "part_id": "2529",
    "available": false,
    "incompatibility": {
      "2530": "Reason for incompatibility"
    }
  }
]
```

각 객체의 필드 설명:
- **part_id**: 부품의 고유 식별자 (문자열)
- **available**: 호환성 여부 (true=호환됨, false=호환되지 않음)
- **incompatibility**: 
  - 호환되는 경우: 빈 객체 `{}`
  - 호환되지 않는 경우: 비호환 부품 ID를 키로, 이유를 값으로 하는 객체

### 5. 호환성 엣지 추출

LLM 응답에서 비호환 관계를 추출하여 "엣지(edge)"라는 구조로 변환합니다:

```json
[
  {
    "part_ids": ["2529", "2530"],
    "reason": "Reason for incompatibility"
  }
]
```

### 6. 시스템 호환성 그래프 업데이트

추출된 호환성 엣지 정보를 사용하여 해당 시스템의 호환성 그래프를 업데이트합니다. 이 정보는 Supabase 데이터베이스의 `systems` 테이블에 저장됩니다.

### 7. API 응답 반환

최종적으로 다음 구조의 응답을 반환합니다:

```json
{
  "parts": [...],  // 부품 정보
  "subsystems": [...],  // 서브시스템 정보
  "systems": [...],  // 시스템 정보
  "compatibility_edges": [...],  // 호환성 엣지
  "compatibility_results": [...]  // LLM 직접 응답
}
```

## 테스트 결과

### 테스트 1: 부품 정보 조회 (/api/debug)

```
curl -X POST https://check-compatibility-api.vercel.app/api/debug -H "Content-Type: application/json" -d '{"part_ids": ["2528", "2529", "2530"]}'
```

**결과**: 성공적으로 부품, 서브시스템, 시스템 정보를 반환했습니다.

### 테스트 2: 호환성 검사 (/api/compatibility-check)

```
curl -X POST https://check-compatibility-api.vercel.app/api/compatibility-check -H "Content-Type: application/json" -d '{"part_ids": ["2528", "2529", "2530"]}'
```

**결과**: 에러 발생
```
{"detail":"Error processing compatibility check: GenerativeModel.__init__() got an unexpected keyword argument 'system_instruction'"}
```

## 문제점 및 개선 방안

### 1. Google Generative AI 라이브러리 호환성 문제

현재 코드는 `system_instruction` 인수를 사용하고 있지만, 배포된 환경에서는 해당 기능을 지원하지 않는 버전의 라이브러리가 사용되고 있습니다.

**상세 분석**:
- 로컬 개발 환경: Google Generative AI 라이브러리 버전 0.7.0 (시스템 인스트럭션 지원)
- 배포 환경: Google Generative AI 라이브러리 버전 0.4.0 (`requirements.txt`에 명시)
- 버전 0.4.0에서는 `system_instruction` 인수가 지원되지 않음

**해결 방안**:

1. **라이브러리 버전 업데이트**:
   ```
   # requirements.txt
   google-generativeai==0.7.0  # 또는 system_instruction을 지원하는 최신 버전
   ```

2. **대체 구현 방법**:
   버전 0.4.0을 계속 사용해야 한다면, 다음과 같이 코드를 수정해야 합니다:
   ```python
   # 시스템 인스트럭션을 프롬프트에 포함시키는 방식
   async def check_compatibility_with_llm(parts_info, subsystems, systems):
       system_instruction = """**Goal:**
       You must evaluate and verify the compatibility of mechanical parts...
       """
       
       input_data = format_llm_input(parts_info, subsystems, systems)
       
       # 시스템 인스트럭션과 입력 데이터를 하나의 프롬프트로 결합
       prompt = f"{system_instruction}\n\nAnalyze the compatibility of the following parts data: {input_data}"
       
       # system_instruction 없이 모델 초기화
       model = genai.GenerativeModel(model_name=GOOGLE_MODEL_NAME)
       
       try:
           # 단일 프롬프트 사용
           response = model.generate_content(prompt)
           # ... 나머지 코드
   ```

3. **safety_settings 및 generation_config 조정**:
   구버전에서 최적의 결과를 얻기 위해 안전 설정과 생성 구성을 조정할 수 있습니다:
   ```python
   safety_settings = [
       {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
       {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
       {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
       {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
   ]
   
   generation_config = {
       "temperature": 0.4,
       "top_p": 0.8,
       "top_k": 40,
       "max_output_tokens": 2048,
   }
   
   model = genai.GenerativeModel(
       model_name=GOOGLE_MODEL_NAME,
       safety_settings=safety_settings,
       generation_config=generation_config
   )
   ```

### 2. 에러 핸들링 개선

현재는 에러가 발생하면 상세 정보가 노출됩니다.

**해결 방안**:
- 보안을 위해 에러 메시지를 추상화하고, 로깅 시스템을 강화해야 합니다.
- 세부 에러 정보는 로그에만 기록하고, 사용자에게는 일반적인 메시지만 반환합니다.

### 3. 입력 유효성 검사 강화

ID 형식 변환은 잘 구현되어 있지만, 잘못된 형식의 ID에 대한 처리가 불충분합니다.

**해결 방안**:
- 더 엄격한 유효성 검사와 명확한 에러 메시지를 제공해야 합니다.
- 존재하지 않는 ID에 대한 처리 로직을 개선합니다.

## 결론

부품 호환성 API는 복잡한 부품 호환성 분석을 자동화하는 강력한 도구입니다. Google Gemini와 같은 LLM을 활용하여 다양한 부품 간의 호환성을 검사하고, 그 결과를 구조화된 형식으로 제공합니다. 

현재 발견된 라이브러리 호환성 문제는 라이브러리 버전을 업데이트하거나 대체 구현 방법을 사용하여 해결할 수 있습니다. 이 문제가 해결되면 API는 매우 유용한 서비스가 될 것입니다.

## 부록: 코드 수정 예시

아래는 Google Generative AI 라이브러리 0.4.0 버전에서 작동하도록 수정한 `check_compatibility_with_llm` 함수 예시입니다:

```python
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
    # 시스템 인스트럭션을 프롬프트의 일부로 포함
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
    
    # 전체 프롬프트 생성 (시스템 인스트럭션 + 입력 데이터)
    prompt = f"{system_instruction}\n\nAnalyze the compatibility of the following parts data: {input_data}"
    
    # 성능 최적화를 위한 설정
    generation_config = {
        "temperature": 0.4,  # 낮은 온도로 일관된 응답 생성
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2048,
    }

    # 기본 모델 초기화 (system_instruction 사용하지 않음)
    model = genai.GenerativeModel(
        model_name=GOOGLE_MODEL_NAME,
        generation_config=generation_config
    )
    
    try:
        # 단일 프롬프트 사용
        response = model.generate_content(prompt)
        print(f"Raw LLM response: {response.text}")
        
        # 나머지 응답 처리 로직은 유지
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
        # 에러 메시지 추상화
        raise Exception(f"Error in LLM compatibility check: Please check logs for details")
```