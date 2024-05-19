from document_processor import load_and_process_documents
import os

# list ur pdfs in a folder and pass the path to the function
#example 
pdf_path = os.listdir('/path/to/your/pdfs/folder')
pdf_path = ['/path/to/your/pdfs/folder/' + path for path in pdf_path if path.endswith('.pdf')]
load_and_process_documents(pdf_path)


