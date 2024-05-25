import json
import os

# Load JSON data
with open('/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/test/all_extracted_data.json', 'r') as file:
    json_data = json.load(file)

directory = './documents_aditya_table'
if not os.path.exists(directory):
    os.makedirs(directory)
# Iterate over each table entry
for pdf_name, pdf_content in json_data.items():
    # Create a directory for each PDF document
    pdf_directory = os.path.join(directory, pdf_name)
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
    for table_entry in pdf_content["tables"]:
        page_number = table_entry["page_number"]
        table_number = table_entry["table_number"]
        data = json.loads(table_entry["data"])

        # Extract columns, index, and data
        columns = data["columns"]
        index = data["index"]
        table_data = data["data"]

        # Create a directory to store table data
        directory = f'./table_data/page_{page_number}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write table data to a file
        table_filename = os.path.join(directory, f'table_{table_number}.txt')
        with open(table_filename, 'w') as table_file:
            table_file.write(f"Columns: {columns}\n")
            table_file.write(f"Index: {index}\n")
            table_file.write("Table Data:\n")
            for row in table_data:
                table_file.write(', '.join(row) + '\n')
