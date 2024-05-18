from document_processor import load_and_process_documents
from retriever_initializer import initialize_retriever
from response_generator import generate_response
from langchain_community.vectorstores import FAISS
from embedding import get_embeddingmodel

pdf_path = '../tata.pdf'
db = FAISS.load_local('./fiass_index_tata', get_embeddingmodel()  , allow_dangerous_deserialization= True)
retriever = initialize_retriever(db)

while True:
    generate_response(retriever, input('\nEnter your query: '))
    
# print("\nAI response:\n", response)
