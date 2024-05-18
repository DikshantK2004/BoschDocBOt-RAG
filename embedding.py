
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def get_embeddingmodel():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")