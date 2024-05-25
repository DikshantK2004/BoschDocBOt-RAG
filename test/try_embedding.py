import json
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import pandas as pd

# Initialize Qdrant client
API_KEY = "47RB2yEaTAKRR-eTXUO4UqpKq8SkzTOnQkIEN-5H0JTGWKGXQLj4Aw"
URL = "https://6bf9b770-63e6-482c-a993-5b0d87dd1c0e.us-east4-0.gcp.cloud.qdrant.io:6333"
qdrant_client = QdrantClient(url=URL, api_key=API_KEY, timeout=1000)

# Load pre-trained models
text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class CLIPEmbedding:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy()

image_embed_model = CLIPEmbedding()

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

image_data = load_json('path/to/images.json')
pdf_data = load_json('path/to/pdf_data.json')

# Process and store image embeddings
image_embeddings = []
for entry in image_data['images']:
    image_path = entry['image_path']
    if os.path.exists(image_path):
        embedding = image_embed_model.embed_image(image_path)[0].tolist()
        image_embeddings.append({
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {"type": "image", "content": image_path}
        })

if not qdrant_client.collection_exists("image_collection"):
    qdrant_client.recreate_collection(
        collection_name="image_collection",
        vectors_config=VectorParams(size=len(image_embeddings[0]['vector']), distance=Distance.COSINE)
    )

qdrant_client.upsert(
    collection_name="image_collection",
    points=image_embeddings
)

# Process and store text and table embeddings
text_embeddings = []

for pdf_file, content in pdf_data.items():
    # Process text data
    for text_entry in content.get('text', []):
        page_number = text_entry['page_number']
        text = text_entry['text']
        if text:
            sentences = text.split('\n')
            embeddings = text_embed_model.encode(sentences)
            for idx, embedding in enumerate(embeddings):
                text_embeddings.append({
                    "id": str(uuid.uuid4()),
                    "vector": embedding.tolist(),
                    "payload": {"type": "text", "content": sentences[idx], "pdf_file": pdf_file, "page_number": page_number}
                })
    
    # Process table data
    for table_entry in content.get('tables', []):
        page_number = table_entry['page_number']
        table_number = table_entry['table_number']
        table_data = table_entry['data']
        df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
        table_strings = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).tolist()
        embeddings = text_embed_model.encode(table_strings)
        for idx, embedding in enumerate(embeddings):
            text_embeddings.append({
                "id": str(uuid.uuid4()),
                "vector": embedding.tolist(),
                "payload": {"type": "table", "content": table_strings[idx], "pdf_file": pdf_file, "page_number": page_number, "table_number": table_number}
            })

if not qdrant_client.collection_exists("text_collection"):
    qdrant_client.recreate_collection(
        collection_name="text_collection",
        vectors_config=VectorParams(size=len(text_embeddings[0]['vector']), distance=Distance.COSINE)
    )

qdrant_client.upsert(
    collection_name="text_collection",
    points=text_embeddings
)

print("Embeddings for images, text, and tables have been generated and stored in Qdrant successfully.")
