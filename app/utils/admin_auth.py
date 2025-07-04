# app/utils/admin_auth.py

from fastapi import Header, HTTPException
from app.core.config import ADMIN_KEY

def verify_admin_key(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True
