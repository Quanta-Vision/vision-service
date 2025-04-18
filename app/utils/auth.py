from fastapi import Header, HTTPException
from app.core.config import BACKEND_SERVICE_KEY

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != BACKEND_SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
