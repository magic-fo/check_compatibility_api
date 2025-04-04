# 부품 호환성 검사 API

이 프로젝트는 부품 간의 호환성을 자동으로 평가하고 호환성 그래프를 생성하는 API를 제공합니다. Google의 Gemini 모델을 활용하여 부품 간의 호환성을 분석하고, 결과를 Supabase 데이터베이스에 저장합니다.

## 최근 업데이트 내용

- **2025-04-04**: Gemini 모델에 전달되는 쿼리를 간소화하여 토큰 사용량 감소 및 처리 효율성 향상
- **2025-03-31**: 초기 버전 릴리스

## 기능

- 여러 부품 ID를 입력받아 호환성 평가 수행
- Google Gemini 2.0 Flash Thinking 모델을 활용한 고급 호환성 분석
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
GEMINI_API_KEY=your-google-api-key

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
{
  "status": "success",
  "message": "Compatibility graph updated successfully",
  "system_id": "5001"
}
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

## LLM 활용 방식

이 API는 호환성 검사를 위해 Google의 Gemini 2.0 Flash Thinking 모델을 사용합니다. 

- 시스템 인스트럭션: 호환성 판단 기준, 응답 포맷, 고려해야 할 물리적/기능적/전기적/소재적 호환성 조건 등을 상세히 정의
- 간소화된 쿼리: 모델에게 "Check the compatibility between the parts provided"와 함께 구조화된 부품 데이터만 전달하여 효율적인 처리 지원

## API 문서

FastAPI가 자동으로 생성한 API 문서는 다음 URL에서 확인할 수 있습니다:

- Swagger UI: `/docs`
- ReDoc: `/redoc`