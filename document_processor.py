from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from embedding import get_embeddingmodel

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

def process_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embedding_function = get_embeddingmodel()
    db = FAISS.from_documents(docs, embedding_function)
    db.save_local('./fiass_index_tata')

def load_and_process_documents(pdf_path):
    from pdf_extractor import extract_text_from_pdf
    
    document_text = extract_text_from_pdf(pdf_path)
    documents = [Document(document_text)]
    return process_documents(documents)
