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
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return docs, embedding_function


def load_and_process_documents(pdf_paths):
    from pdf_extractor import extract_text_from_pdf
    all_documents = []
    embedding_function = None  # Initialize the embedding function variable

    # Initialize an empty FAISS index
    faiss_index = None

    for pdf_path in pdf_paths:
        # Extract text from each PDF and create a Document object
        document_text = extract_text_from_pdf(pdf_path)
        documents = [Document(document_text)]
        all_documents.extend(documents)

        # Process documents to get text chunks and load embeddings model only once
        if embedding_function is None:
            docs, embedding_function = process_documents(documents)
        else:
            docs, _ = process_documents(documents)

        # Create or merge documents into the FAISS index
        if faiss_index is None:
            faiss_index = FAISS.from_documents(docs, embedding_function)
        else:
            faiss_index_i = FAISS.from_documents(docs, embedding_function)
            faiss_index.merge_from(faiss_index_i)

    # Save the fully merged index after all documents are processed
    faiss_index.save_local('./faiss_index')
