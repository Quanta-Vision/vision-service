# app/utils/consumer.py
from fastapi import Request, HTTPException

def get_consumer(request: Request):
    consumer = getattr(request.state, "consumer_name", None)
    if not consumer:
        raise HTTPException(status_code=401, detail="Consumer info missing")
    return consumer
