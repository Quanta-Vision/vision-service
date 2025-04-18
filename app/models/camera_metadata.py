from pydantic import BaseModel
from typing import Optional

class Metadata(BaseModel):
    camera_id: str
    location: str
    zone: Optional[str] = None
