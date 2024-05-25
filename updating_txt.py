import os

def append_text_to_files(source_directory, target_directory, mapping):
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    lower_mapping = {key.lower(): value for key, value in mapping.items()}
    # Walk through all directories in the source directory
    for subdir, dirs, files in os.walk(source_directory):
        car_name = os.path.basename(subdir).lower()  # Get the directory name and convert to lower case
        print(car_name)
        # Check if the directory name matches any car in the mapping (also in lower case)
        matched_car_name = next((car for car in lower_mapping if car in car_name), None)

        if matched_car_name:
            # Process each file within the directory
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(subdir, file)
                    target_file_path = os.path.join(target_directory, os.path.basename(subdir), file)

                    # Ensure the target subdirectory exists
                    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

                    # Find the company using the mapping
                    company = lower_mapping[matched_car_name]  # Convert back to title case to match keys
                    
                    # Text to prepend
                    text_to_prepend = f"This content is extracted from the owner's manual of {matched_car_name}, "\
                                      f"which is a model of {company} company.\n\n"

                    # Read the existing content from the source file
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # Prepend the text and write to the new file
                    with open(target_file_path, 'w') as f:
                        f.write(text_to_prepend + content)
                    
                    print(f"Updated file written to {target_file_path}")

# Base and target directories
base_directory = '/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/documents_aditya'
target_directory = '/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/updated_documents'

# Mapping of car names to companies
car_company_mapping = {
    'Nexon': 'Tata',
    'Punch': 'Tata',
    'Next_Gen_Verna': 'Hyundai',
    'Exter': 'Hyundai',
    # Add more mappings as necessary
}

# Call the function with the base directory, target directory, and car-company mapping
append_text_to_files(base_directory, target_directory, car_company_mapping)
