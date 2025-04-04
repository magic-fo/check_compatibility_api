import sys
import os
from fastapi import FastAPI
from mangum import Mangum

# 디버그 정보 출력
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Available environment variables: {list(os.environ.keys())}")

app = FastAPI()

@app.get("/debug/test")
def test_endpoint():
    return {
        "status": "ok", 
        "message": "Debug endpoint is working!",
        "python_version": sys.version,
        "environment": os.environ.get("VERCEL_ENV", "unknown")
    }

# 핸들러 함수 - 이 부분이 Vercel 서버리스 함수와 연결됩니다
handler = Mangum(app) 