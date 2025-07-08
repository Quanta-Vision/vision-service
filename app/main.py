import uvicorn
from app.api.routes_iam import router_iam
from app.api.routes import router
from app.api.routes_recognite import router_v2
from app.api.routes_counter import router_counter
from fastapi import FastAPI
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

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
