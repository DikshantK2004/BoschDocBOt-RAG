from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader
from qdrant_client.models import Distance, VectorParams
API_KEY = "47RB2yEaTAKRR-eTXUO4UqpKq8SkzTOnQkIEN-5H0JTGWKGXQLj4Aw"
URL = "https://6bf9b770-63e6-482c-a993-5b0d87dd1c0e.us-east4-0.gcp.cloud.qdrant.io:6333"
qdrant_client = QdrantClient(
    url=URL, 
    api_key=API_KEY,
    timeout= 1000
)



from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np


# Text embedding model
text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Image embedding model using CLIP
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


# Create the MultiModal index
documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

for doc in documents:
    print(type(doc))
    if str(type(doc)) == "<class 'llama_index.core.schema.ImageDocument'>":
        l = image_embed_model.embed_image(doc.image_path)
        print(doc.image_path, l.shape)
        doc.embedding = l[0].tolist()
    else:
        l = text_embed_model.encode(doc.text)
        print(l.shape)
        doc.embedding = l.tolist()
        
# props = documents[0].__dict__.keys()

# for prop in props:
#     print(f"Property: {prop}")
#     print(f"Value: {documents[0].__dict__[prop]}")
    
# print("\n\n")
# props = documents[31].__dict__.keys()
# for prop in props:
#     print(f"Property: {prop}")
#     print(f"Value: {documents[30].__dict__[prop]}")
    

# Create a text collection
if  qdrant_client.collection_exists("text_collection") is False:
    qdrant_client.recreate_collection(
        collection_name="text_collection",
        vectors_config=VectorParams(
            size=len(documents[31].embedding), distance=Distance.COSINE
        )
    )

# Create an image collection
if qdrant_client.collection_exists("image_collection") is False:
    qdrant_client.recreate_collection(
        collection_name="image_collection",
        vectors_config=VectorParams(
            size=len(documents[0].embedding), distance=Distance.COSINE
        )
    )

from llama_index.core.schema import Document, ImageDocument
import uuid

def store_documents(documents):
    text_points = []
    image_points = []
    
    for doc in documents:
        if str(type(doc)) == "<class 'llama_index.core.schema.Document'>":
            text_point = {
                "id": str(uuid.uuid4()),
                "vector": doc.embedding,
                "payload": {"type": "text", "content": doc.text}
            }
            text_points.append(text_point)
        else:
            image_point = {
                "id": str(uuid.uuid4()),
                "vector": doc.embedding,
                "payload": {"type": "image", "content": doc.image_path}
            }
            image_points.append(image_point)
    
    if text_points:
        qdrant_client.upsert(
            collection_name="text_collection",
            points=text_points
        )
    
    if image_points:
        qdrant_client.upsert(
            collection_name="image_collection",
            points=image_points
        )

store_documents(documents)
