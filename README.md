# vision-service

**vision-service** is a FastAPI-based backend service designed to process images for facial recognition, specifically tailored for roll call systems (e.g., in schools, workplaces). It allows you to add people by uploading their images and later recognize individuals by submitting a photo.

The service uses **FaceNet** (via `facenet-pytorch`) to extract facial embeddings and matches people via cosine similarity. It supports storing data locally in development or in cloud (GCP/AWS) in production environments.

---

## ✨ Features

- Add a person with one or more images
- Add many people at once (bulk endpoint coming soon)
- Recognize a person from a given image
- Save and retrieve person data from MongoDB
- Save image & model files to local or cloud storage

---

## 📈 Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- MongoDB (as database)
- Local/Cloud Storage (GCP or AWS)
- Python 3.8+

---

## 📚 Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Locally

### 1. Clone the project
```bash
git clone https://github.com/your-username/vision-service.git
cd vision-service
```

### 2. Create `.env` file

```env
APP_ENV=development
PORT=8000
MONGODB_URL=mongodb://localhost:27017
CLOUD_PROVIDER=aws   # or gcp
```

### 3. Start the API
```bash
python -m app.main
```

> Access Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Using Uvicorn CLI (alternative)
```bash
uvicorn app.main:app --reload --port $PORT
```

If on Windows PowerShell:
```powershell
$env:PORT=8000
uvicorn app.main:app --reload --port $env:PORT
```

---

## 🎭 API Endpoints

### POST `/add-person`
Add a single person

**Form fields:**
- `name`: string
- `user_id`: string
- `images`: list of image files (jpg, png)

### POST `/recognize`
Recognize a person from an uploaded image

**Form field:**
- `image`: single image file

---

## 🌐 Environment Behavior

| Variable     | Description                         |
|--------------|-------------------------------------|
| `APP_ENV`    | `development` or `production`       |
| `PORT`       | The port FastAPI will run on        |
| `MONGODB_URL`| MongoDB connection string           |
| `CLOUD_PROVIDER` | `aws` or `gcp` (used in prod)   |

- When `APP_ENV=development`, images and models are stored in local folders.
- When `APP_ENV=production`, files will be stored to GCP or AWS bucket.

---

## ⚒️ TODO
- [ ] Add `/add-many-persons` bulk upload API
- [ ] Support for GCP & AWS cloud uploads
- [ ] Dockerfile for deployment

---

## 🚀 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## ✅ License
[MIT](LICENSE)

**Typical InsightFace distances (when using buffalo_l and L2-norm):**
+ Same person, same pose: often < 1.0
+ Same person, different pose/lighting/glasses: can be up to 2.0 – 4.0
+ Different people: often > 5.0, but depends on registration quality
