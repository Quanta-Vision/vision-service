import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
from torchvision import transforms
from scipy.spatial.distance import cosine

# Load pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def extract_face_embedding(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 160, 160)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()  # shape: (512,)
    return embedding

def find_best_match(unknown_embedding, people):
    best_match = None
    best_score = 0.0
    threshold = 0.6

    for person in people:
        for emb in person.get("embeddings", []):
            score = 1 - cosine(unknown_embedding, np.array(emb))
            if score > best_score and score > (1 - threshold):
                best_score = score
                best_match = person

    return best_match
