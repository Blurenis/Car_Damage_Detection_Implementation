from huggingface_hub import InferenceClient
from PIL import Image
import requests
from io import BytesIO

# Charger l'image
url = "https://your_image_url.jpg"
image = Image.open(BytesIO(requests.get(url).content))

# Client API (tu dois avoir un token HF : https://huggingface.co/settings/tokens)
client = InferenceClient(token="hf_xxx")

# Appel du modèle (remplace par le repo correct sur HF)
result = client.image_segmentation(model="intelliarts/Car_damage_detection", image=image)

print(result)