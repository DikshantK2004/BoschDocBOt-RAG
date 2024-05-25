from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader
from qdrant_client.models import Distance, VectorParams, PointStruct


qdrant_client = QdrantClient(
    timeout=3000,
    path='qdrant_tata'
)



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


# Create the MultiModal index
text_documents = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/documents_aditya/Next_Gen_Verna.pdf").load_data()
image_documents = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/content/verna/images_verna").load_data()


print(len(text_documents))
print(len(image_documents))
text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')


for doc in text_documents:
        l = text_embed_model.encode(doc.text)
        # print(l.shape)
        doc.embedding = l.tolist()

for img_doc in image_documents:
    img_doc.embedding = image_embed_model.embed_image(img_doc.image_path)[0].tolist()
    # print(img_doc.embedding.shape)
    
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
    qdrant_client.create_collection(
        collection_name="text_collection",
        vectors_config=VectorParams(
            size=len(text_documents[0].embedding), distance=Distance.COSINE
        )
    )

# Create an image collection
if qdrant_client.collection_exists("image_collection") is False:
    qdrant_client.create_collection(
        collection_name="image_collection",
        vectors_config=VectorParams(
            size=len(image_documents[0].embedding), distance=Distance.COSINE
        )
    )

from llama_index.core.schema import Document, ImageDocument
import uuid

def store_documents(text_documents,image_documents):
    text_points = []
    image_points = []
    
    for doc in text_documents:
            text_point = PointStruct(
                id=  str(uuid.uuid4()),
                vector=  doc.embedding,
                payload=  {"type": "text", "content": doc.text},
            )
            text_points.append(text_point)
    
    for doc in image_documents:
            image_point = PointStruct(
                id = str(uuid.uuid4()),
                vector= doc.embedding,
                payload= {"type": "image", "content": doc.image_path},
            )
            image_points.append(image_point)
    
    if text_points:
        qdrant_client.upsert(
            collection_name="text_collection",
            points=text_points
        )
    
    # if image_points:
    #     qdrant_client.upsert(
    #         collection_name="image_collection",
    #         points=image_points
    #     )

store_documents(text_documents, image_documents)
