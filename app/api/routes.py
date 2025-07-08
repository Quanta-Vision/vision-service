from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["System Health"], summary="Health Check")
def health_check():
    """Returns status of the vision service."""
    return {"status": "ok", "message": "vision service is running"}
