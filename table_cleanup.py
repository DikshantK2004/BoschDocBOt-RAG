import os
import shutil

# Function to remove files containing 'NOTE' or empty 'Table Data:'
def remove_unwanted_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'NOTE' in content or ('Table Data:' in content and content.strip().endswith('Table Data:')):
                        os.remove(file_path)

# Function to rename folders
def rename_folders(base_path, rename_mapping):
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            new_name = rename_mapping.get(dir_name, dir_name)
            old_path = os.path.join(root, dir_name)
            new_path = os.path.join(root, new_name)
            if old_path != new_path:
                shutil.move(old_path, new_path)

# Function to remove empty folders
def remove_empty_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

# Path to the extracted documents
extracted_folder_path = '/home/dikshant/Downloads/documents_aditya_table/documents_aditya_table/'

# # Remove files containing 'NOTE' or empty 'Table Data:'
# remove_unwanted_files(extracted_folder_path)

# Rename folders
rename_mapping = {
    'punch-bsvi-09-09-21.pdf': 'punch',
    'nexon-owner-manual-2022.pdf': 'nexon',
    'Next_Gen_Verna.pdf': 'verna',
    'exter.pdf': 'exter'
}

rename_folders(extracted_folder_path, rename_mapping)

# Remove empty folders
remove_empty_folders(extracted_folder_path)

# Verify the changes
updated_structure = []
for root, dirs, files in os.walk(extracted_folder_path):
    for file in files:
        updated_structure.append(os.path.join(root, file))

# Display updated structure
print("Updated File Structure:")
for path in updated_structure:
    print(path)
