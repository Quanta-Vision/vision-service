from fastapi import Header, HTTPException, Depends, Request
from app.models.api_keys import find_api_key

async def verify_api_key(
    x_api_key: str = Header(...),
    request: Request = None
):
    api_key_doc = find_api_key(x_api_key)
    if not api_key_doc:
        raise HTTPException(status_code=401, detail="Unauthorized (invalid or inactive API key)")
    # Optionally check allowed_apis:
    api = request.url.path
    allowed = api_key_doc.get("allowed_apis", ["*"])
    if "*" not in allowed and api not in allowed:
        raise HTTPException(status_code=403, detail="Forbidden for this API")
    # Attach consumer_name to request.state if you want
    if request:
        request.state.consumer_name = api_key_doc.get("consumer_name")
    return True
