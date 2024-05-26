from qdrant_client import QdrantClient , models
from typing import List

qdrant_client = QdrantClient(path = "./qdrant_data_fresh_5",
                      timeout   = 3000)

from pickle import load
summaries = []
with open('summarized_tables.pkl', 'rb') as f:
    summaries = load(f)

from sentence_transformers import SentenceTransformer

text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
table_points  : List[models.PointStruct]= []
import uuid
print(len(summaries))
for summary in summaries:
    table_point = models.PointStruct(
        id = str(uuid.uuid4()),
        vector = text_embed_model.encode(summary ),
        payload = {"type": "table", "content": summary}
    )
    table_points.append(table_point)

    
    
text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
if qdrant_client.collection_exists(f"table_collection") is False:
        qdrant_client.create_collection(
            collection_name=f"table_collection",
            vectors_config=models.VectorParams(
                size= len(table_points[0].vector), distance=models.Distance.COSINE
            )
        )

    
qdrant_client.upsert(collection_name=f"table_collection", points=table_points)

