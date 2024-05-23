from qdrant_client import QdrantClient
from qdrant_client.http.models import ScrollRequest

# Initialize Qdrant client
client = QdrantClient(path="qd2")

def query_all_documents(collection_name):
    results = []
    scroll_params = {
        'limit': 100
    }

    while True:
        scroll_result = client.scroll(collection_name=collection_name, **scroll_params)
        print(scroll_result)
        if scroll_result.next_page_offset is None:
            break
        scroll_params['offset'] = scroll_result.next_page_offset

    return results

# Query text collection
text_documents = query_all_documents("text_collection")
print("Text Documents:")
for doc in text_documents:
    print(doc.payload)

# Query image collection
image_documents = query_all_documents("image_collection")
print("Image Documents:")
for doc in image_documents:
    print(doc.payload)
