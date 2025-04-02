from app.core.config import APP_ENV, CLOUD_PROVIDER
import os
import uuid

def save_image(file, person_id):
    ext = file.filename.split(".")[-1]
    filename = f"{person_id}_{uuid.uuid4()}.{ext}"

    if APP_ENV == "development":
        os.makedirs("images", exist_ok=True)
        path = f"images/{filename}"
        with open(path, "wb") as f:
            f.write(file.file.read())
        return path

    if APP_ENV == "production":
        if CLOUD_PROVIDER == "aws":
            # Upload to AWS S3
            pass
        elif CLOUD_PROVIDER == "gcp":
            # Upload to GCP
            pass
