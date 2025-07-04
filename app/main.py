import uvicorn
from app.api.routes import router
from app.api.routes_v2 import router_v2
from fastapi import FastAPI
from app.core.config import PORT

app = FastAPI(title="Vision API")
app.include_router(router)
app.include_router(router_v2, prefix="/v2")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
