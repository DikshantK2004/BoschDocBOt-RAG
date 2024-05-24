from qdrant_client import QdrantClient

client = QdrantClient(path = "./qdrant_tata",
                      timeout= 3000)



from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np


# Text embedding model
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
        # print(outputs.cpu().numpy())
        return outputs.cpu().numpy()
    
    def embed_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs.cpu().numpy()

image_embed_model = CLIPEmbedding()

text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')


query = "What is vehicle identification number?"
resp = client.search(collection_name='text_collection', query_vector= text_embed_model.encode(query) , limit=5)

props = resp[0].__dict__.keys()

for r in resp:
    print('--------------------------------------------------------')
    print(r.score)
    print(r.payload['content'])

resp = client.search(collection_name='image_collection', query_vector= image_embed_model.embed_text(query)[0] , limit=5)
print('--------------------------------------------------------')
print('--------------------------------------------------------')

props = resp[0].__dict__.keys()

for r in resp:
    print('--------------------------------------------------------')
    print(r.score)
    print(r.payload['content'])