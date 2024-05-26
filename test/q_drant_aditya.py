from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
import re
qdrant_client = QdrantClient(
    timeout=3000,
    path='qdrant_data_why_new_model_now'
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

cur_dir = os.getcwd()

# Create the MultiModal index
<<<<<<< HEAD
# text_documents_1 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/test/updated_documents/Next_Gen_Verna.pdf").load_data()
# text_documents_2 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/test/updated_documents/exter.pdf").load_data()
# text_documents_3 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/test/updated_documents/nexon-owner-manual-2022.pdf").load_data()
text_documents_4 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/documents_aditya/exter.pdf").load_data()
=======
text_documents_1 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/updated_documents/Next_Gen_Verna.pdf").load_data()
text_documents_2 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/updated_documents/exter.pdf").load_data()
text_documents_3 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/updated_documents/nexon-owner-manual-2022.pdf").load_data()
text_documents_4 = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/updated_documents/punch-bsvi-09-09-21.pdf").load_data()
>>>>>>> 3b35d0cf8ee135febd39cba6fdb866cec0647182

text_documents = text_documents_1 + text_documents_2 + text_documents_3 + text_documents_4

image_documents = SimpleDirectoryReader("/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/test/images").load_data()

text_data = [doc.to_dict() for doc in text_documents]  # Convert each document to dictionary if not already
image_data = [doc.to_dict() for doc in image_documents]

# with open('text_documents.json', 'w') as f:
#     json.dump(text_data, f)

with open('image_documents.json', 'w') as f:
    json.dump(image_data, f)

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
<<<<<<< HEAD
if  qdrant_client.collection_exists("text_collection_exter") is False:
    qdrant_client.create_collection(
        collection_name="text_collection_exter",
=======
if  qdrant_client.collection_exists("text_collection") is False:
    qdrant_client.create_collection(
        collection_name="text_collection",
>>>>>>> 3b35d0cf8ee135febd39cba6fdb866cec0647182
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

# from llama_index.core.schema import Document, ImageDocument
# import uuid

# def extract_pdf_info_regex(file_path):
#     # Define a regular expression pattern to extract the PDF name and page number
#     pattern = r'.*/([^/]+)_page_(\d+)\.txt$'
#     match = re.search(pattern, file_path)
    
#     if match:
#         pdf_name = match.group(1)  # 'Next_Gen_Verna.pdf'
#         page_number = match.group(2)  # '1'
#         print(pdf_name, page_number)
#         return pdf_name, page_number
#     else:
#         return None, None
    
# def extract_pdf_info_regex_image(file_path):
#     # Define a regular expression pattern to extract the PDF name and page number
#     pattern = r'([^/]+)_page_(\d+)_image_(\d+)\.png$'
#     match = re.search(pattern, file_path)
    
#     if match:
#         pdf_name = match.group(1)  # 'Next_Gen_Verna.pdf'
#         page_number = match.group(2)  # '1'
#         print(pdf_name, page_number)
#         return pdf_name, page_number
#     else:
#         return None, None

def store_documents(text_documents,image_documents):
    text_points = []
    image_points = []
    
    for doc in text_documents:
            data = doc.metadata
            print(data['file_path'])
            pdf_name, page_number = extract_pdf_info_regex(data['file_path'])
            if pdf_name is not None and page_number is not None:
                text_point = PointStruct(
                    id=  str(uuid.uuid4()),
                    vector=  doc.embedding,
                    payload=  {"type": "text", "content": doc.text, "page": int(page_number), "pdf_name": pdf_name},
                )
            else :
                  text_point = PointStruct(
                    id=  str(uuid.uuid4()),
                    vector=  doc.embedding,
                    payload=  {"type": "text", "content": doc.text},
                )
            text_points.append(text_point)
    
    for doc in image_documents:
            data = doc.metadata
            print(data['file_path'])
            pdf_name, page_number = extract_pdf_info_regex_image(data['file_path'])
            if pdf_name is not None and page_number is not None:
                image_point = PointStruct(
                    id=  str(uuid.uuid4()),
                    vector=  doc.embedding,
                    payload=  {"type": "text", "content": doc.image_path, "page": int(page_number), "pdf_name": pdf_name},
                )
            else :
                  image_point = PointStruct(
                    id=  str(uuid.uuid4()),
                    vector=  doc.embedding,
                    payload=  {"type": "text", "content": doc.image_path},
                )
            image_points.append(image_point)
    
    if text_points:
        qdrant_client.upsert(
<<<<<<< HEAD
            collection_name="text_collection_exter",
=======
            collection_name="text_collection",
>>>>>>> 3b35d0cf8ee135febd39cba6fdb866cec0647182
            points=text_points
        )
    
    if image_points:
        qdrant_client.upsert(
            collection_name="image_collection",
            points=image_points
        )

store_documents(text_documents, image_documents)
