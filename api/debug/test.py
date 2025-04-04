from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/debug/test")
def test_endpoint():
    return {"status": "ok", "message": "Debug endpoint is working!"}

# 핸들러 함수 - 이 부분이 Vercel 서버리스 함수와 연결됩니다
handler = Mangum(app) 