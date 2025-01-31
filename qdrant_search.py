from qdrant_client import QdrantClient , models

client = QdrantClient(path = "./qdrant_data_fresh_5",
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

rs = []
query = input("Enter the query: ")
for pdf in ["nexon", "verna", "punch", "exter"]:
    resp = client.search(collection_name=f'text_collection_{pdf}', query_vector= text_embed_model.encode(query) , limit=2)
    rs += resp

rs += client.search(collection_name='table_collection', query_vector= text_embed_model.encode(query) , limit=2)
# props = resp[0].__dict__.keys()

for r in rs:
    print('--------------------------------------------------------')
    # print(r.score)
    print(r.payload['content'])
    # print(r.payload['page'])
    # print(r.payload['pdf_name'])


    # pdf_name = r.payload['pdf_name'].replace('.pdf','')
    # print(pdf_name)
    # page = r.payload['page']
    # query_filter=models.Filter(
    #     must=[
    #     models.FieldCondition(
    #         key="pdf_name",  # Assuming 'pdf_name' is the field in your collection
    #         match=models.MatchValue(
    #             value=pdf_name # Replace with the specific PDF name you're looking for
    #         ),
    #     ),
    #     models.FieldCondition(
    #         key="page",  # Assuming 'page' is the field in your collection
    #         match=models.MatchValue(
    #             value=page  # Replace with the specific page number you're looking for
    #         ),
    #     )
    # ]
    # )
    # # Perform image search based on text result
    # image_response = client.search(
    #     collection_name='image_collection', 
    #     query_vector=image_embed_model.embed_text(query)[0], 
    #     query_filter=query_filter,
    #     limit=1
    # )

    # print('--------------------------------------------------------')
    # # print('Image results for the above text:')
    # for img in image_response:
    #     print('--------------------------------------------------------')
    #     print(f"Score: {img.score}")
    #     print(f"Image Content: {img.payload['content']}")
    #     Image.open(img.payload['content']).show()

# resp = client.search(collection_name='image_collection', query_vector= image_embed_model.embed_text(query)[0] , limit=5)
# print('--------------------------------------------------------')
# print('--------------------------------------------------------')

# # props = resp[0].__dict__.keys()

# for r in resp:
#     print('--------------------------------------------------------')
#     print(r.score)
#     print(r.payload['content'])