# 부품 호환성 검사 API

이 프로젝트는 부품 간의 호환성을 자동으로 평가하고 호환성 그래프를 생성하는 API를 제공합니다. Google의 Gemini 모델을 활용하여 부품 간의 호환성을 분석하고, 결과를 Supabase 데이터베이스에 저장합니다.

## 기능

- 여러 부품 ID를 입력받아 호환성 평가 수행
- Google Gemini LLM을 활용한 고급 호환성 분석
- 호환성 그래프 생성 및 저장
- Vercel에서의 서버리스 배포 지원

## 기술 스택

- **Web API 프레임워크**: FastAPI
- **LLM API**: Google Gemini
- **데이터베이스**: Supabase
- **배포**: Vercel (서버리스)

## 설치 및 실행

### 로컬 개발

1. 의존성 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:

`.env.example` 파일을 `.env`로 복사하고 필요한 값을 채워넣으세요:

```
# Supabase 연결 정보
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# Google AI 연결 정보
GOOGLE_API_KEY=your-google-api-key
GOOGLE_MODEL_NAME=gemini-2.0-flash-thinking-exp

# 로그 레벨 설정
LOG_LEVEL=INFO
```

3. 애플리케이션 실행:

```bash
uvicorn main:app --reload
```

### Vercel 배포

1. Vercel CLI 설치:

```bash
npm install -g vercel
```

2. 배포 실행:

```bash
vercel
```

3. 환경 변수 설정:

Vercel 대시보드에서 프로젝트에 필요한 환경 변수를 설정하세요.

## API 사용법

### 호환성 검사

**엔드포인트**: `POST /api/compatibility-check`

**요청 예시**:

```http
POST /api/compatibility-check
Content-Type: application/json

{
  "part_ids": [1001, 1002, 1003]
}
```

**응답 예시**:

```json
[
  {
    "part_id": 1001,
    "product_name": "Hydraulic Pump XJ-42",
    "available": true,
    "incompatibility": null
  },
  {
    "part_id": 1002,
    "product_name": "Control Valve CV-101",
    "available": false,
    "incompatibility": {
      "1003": "Thread type mismatch: Control Valve CV-101 uses NPT threads while Pressure Sensor PS-200 uses BSPP threads"
    }
  },
  {
    "part_id": 1003,
    "product_name": "Pressure Sensor PS-200",
    "available": false,
    "incompatibility": {
      "1002": "Thread type mismatch: Pressure Sensor PS-200 uses BSPP threads while Control Valve CV-101 uses NPT threads"
    }
  }
]
```

## 데이터베이스 구조

호환성 그래프는 `systems` 테이블의 `compatibility_graph` 필드에 다음 구조로 저장됩니다:

```json
{
  "edges": [
    {
      "part_ids": [1002, 1003],
      "reason": "Thread type mismatch: Control Valve CV-101 uses NPT threads while Pressure Sensor PS-200 uses BSPP threads"
    }
  ]
}
```

## API 문서

FastAPI가 자동으로 생성한 API 문서는 다음 URL에서 확인할 수 있습니다:

- Swagger UI: `/docs`
- ReDoc: `/redoc`