import time
from fastapi.responses import JSONResponse
import uvicorn
from app.api.routes_iam import router_iam
from app.api.routes import router
from app.api.routes_recognite import router_v2
from app.api.routes_counter import router_counter
from app.api.routes_liveness import router_liveness
from app.api.router_ai_liveness import router_ai_liveness
from fastapi import FastAPI, Request
from app.core.config import PORT
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

os.makedirs("images", exist_ok=True)  # Ensure folder exists

app = FastAPI(title="Vision API")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(router_iam, prefix="/iam")
app.include_router(router_v2, prefix="/v2")
app.include_router(router_counter, prefix="/counter")
app.include_router(router_liveness, prefix="/spoof-detect")
app.include_router(router_ai_liveness, prefix="/ai-spoof-detect")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during liveness detection",
            "timestamp": int(time.time() * 1000)
        }
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
