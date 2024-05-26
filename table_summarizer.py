import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain.schema import StrOutputParser

# Initialize the LLM
print("Initializing the LLM...")
llm = Ollama(model="phi3", callback_manager=CallbackManager([]))
print("LLM Initialized.")

# Function to summarize a table using the LLM
def summarize_table(file_path, pdf_name):
    with open(file_path, 'r') as file:
        text = file.read()

    prompt_template = f"""
    You are a table summarizer bot. I have extracted data of a table. You are supposed to summarize each row into 1 line so that it looks neat.
    Basically, this is from the manual of {pdf_name}. So include that in the summary of each line.
    So, each row should start like, For {pdf_name}.................
    Understand the context by seeing the heading of the table and summarize each row into 1 line.
    Create output only on the table provided. Adding anything of your own means losing job.
    Complete them as sentences. Don't write any notes at all."""

    pr = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{input}"),
        ]
    )

    fin = pr.format(input=text)
    response = llm.generate(prompts=[fin])
    ans = StrOutputParser().parse(response).generations[0][0].text

    return ans

# Path to the extracted documents
extracted_folder_path = './cleaned_table_documents'

# Traverse directories and summarize all tables
summaries = []
total_files = 0

# Count the total number of .txt files first
print("Counting the total number of .txt files...")
for root, dirs, files in os.walk(extracted_folder_path):
    for file in files:
        if file.endswith('.txt'):
            total_files += 1
print(f"Total .txt files to process: {total_files}")

processed_files = 0


for root, dirs, files in os.walk(extracted_folder_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            pdf_name = file_path.split('documents/')[1].split('/')[0]  # Get the PDF name from the folder structure
            print(f"Processing file: {file_path} {pdf_name}...")
            summary = summarize_table(file_path, pdf_name)
            summaries.append((file_path, summary))
            processed_files += 1
            print(f"Processed {processed_files}/{total_files} files.")

# Print all summaries
print("Printing all summaries...")
for file_path, summary in summaries:
    print(f"File: {file_path}\nSummary:\n{summary}\n")

# Optionally, save the summaries to a file
output_file_path = "all_summarized_tables.txt"

sum_files = [summary[1] for summary in summaries]

import pickle

with open('summarized_tables.pkl', 'wb') as f:
    pickle.dump(sum_files, f)
    
print(f"Saving all summaries to {output_file_path}...")
with open(output_file_path, "w") as output_file:
    for file_path, summary in summaries:
        output_file.write(f"File: {file_path}\nSummary:\n{summary}\n\n")
print("All summaries saved.")
