from fastapi import APIRouter, Depends, Form, HTTPException, Body
from typing import Optional, List
from app.models.api_keys import (
    create_api_key,
    deactivate_api_key,
    update_consumer,
    list_all_api_keys,
)
import secrets

from app.utils.admin_auth import verify_admin_key

router_iam = APIRouter()

@router_iam.post("/generate-api-key", tags=["IAM"], dependencies=[Depends(verify_admin_key)])
async def generate_api_key(
    consumer_name: str = Form(...),
    allowed_apis: Optional[str] = Form("*")
):
    api_key = secrets.token_urlsafe(32)
    allowed_api_list = allowed_apis.split(",") if allowed_apis != "*" else ["*"]
    create_api_key(api_key, consumer_name, allowed_api_list)
    return {"api_key": api_key, "consumer_name": consumer_name}

@router_iam.get("/list-api-keys", tags=["IAM"], dependencies=[Depends(verify_admin_key)])
async def list_api_keys():
    keys = list_all_api_keys()
    return {"api_keys": keys, "count": len(keys)}

@router_iam.put("/update-consumer", tags=["IAM"], dependencies=[Depends(verify_admin_key)])
async def update_consumer_api(
    api_key: str = Form(...),
    consumer_name: Optional[str] = Form(None),
    allowed_apis: Optional[str] = Form(None),
    active: Optional[bool] = Form(None),
):
    updates = {}
    if consumer_name is not None:
        updates["consumer_name"] = consumer_name
    if allowed_apis is not None:
        updates["allowed_apis"] = allowed_apis.split(",") if allowed_apis != "*" else ["*"]
    if active is not None:
        updates["active"] = active
    if not updates:
        raise HTTPException(status_code=400, detail="No update fields provided.")
    result = update_consumer(api_key, updates)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="API key not found.")
    return {"msg": "Consumer updated successfully."}

@router_iam.delete("/delete-api-key", tags=["IAM"], dependencies=[Depends(verify_admin_key)])
async def delete_api_key_api(
    api_key: str = Form(...)
):
    result = deactivate_api_key(api_key)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="API key not found.")
    return {"msg": "API key deactivated (deleted) successfully."}
