# app/utils/admin_auth.py

from fastapi import Header, HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED
from app.core.config import ADMIN_KEY, ADMIN_ALLOWED_IPS
import logging

def verify_admin_key(request: Request, x_admin_key: str = Header(...)):
    client_ip = request.client.host
    if client_ip not in ADMIN_ALLOWED_IPS:
        logging.warning(f"Blocked unauthorized IP: {client_ip}")
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized IP address")

    if x_admin_key != ADMIN_KEY:
        logging.warning(f"Invalid admin key from {client_ip}")
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid admin key")
    return True
