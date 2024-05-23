from unstructured.partition.pdf import partition_pdf
# Extract images, tables, and chunk text
raw_pdf_elements = 0

raw_pdf_elements = partition_pdf(
    filename='../resume.pdf',
    # extract_images_in_pdf=True,
    # infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=1000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    # image_output_dir_path=path,
)
