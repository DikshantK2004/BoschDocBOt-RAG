import json 
import os

with open('/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/test/all_extracted_data.json', 'r') as file:
    pdf_data = json.load(file)

# Create a directory to store documents
directory = './documents_aditya'
if not os.path.exists(directory):
    os.makedirs(directory)

# Iterate over the PDF data and save each document as a separate file
for pdf_name, pdf_content in pdf_data.items():
    # Create a directory for each PDF document
    pdf_directory = os.path.join(directory, pdf_name)
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)

    # Extract text content from each page and save it to separate text files
    for page_data in pdf_content['text']:
        page_number = page_data['page_number']
        page_text = page_data['text']
        page_filename = os.path.join(pdf_directory, f'{pdf_name}_page_{page_number}.txt')
        with open(page_filename, 'w') as page_file:
            page_file.write(page_text)

    # Save tables as separate files if available
    if 'tables' in pdf_content:
        for table_data in pdf_content['tables']:
            table_number = table_data['table_number']
            table_content = table_data['data']
            
            # Skip saving empty tables
            if not table_content:
                continue
                
            table_filename = os.path.join(pdf_directory, f'{pdf_name}_table_{table_number}.json')
            with open(table_filename, 'w') as table_file:
                json.dump(table_content, table_file)
