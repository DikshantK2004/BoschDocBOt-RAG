import fitz  # Import the PyMuPDF library
import os

output_folder = 'extracted_images'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist


def extract_images_and_texts(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    for page_number, page in enumerate(doc):
        # Extract images
        image_list = page.get_images(full=True)
        extracted_images = []
        for img_ref in image_list:
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/image_{page_number}_{xref}.{image_ext}"

            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            extracted_images.append({
                "file_path": image_filename,
                "position": base_image["bbox"]  # Bounding box of the image
            })

        # Extract text
        texts = []
        for text_instance in page.get_text("dict")["blocks"]:
            if text_instance['type'] == 0:  # This is a text block
                texts.append({
                    "content": text_instance["text"],
                    "bbox": text_instance["bbox"]
                })

        results.append({
            "page": page_number,
            "images": extracted_images,
            "texts": texts
        })

    return results

pdf_path = '/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/pdfs/punch-bsvi-09-09-21.pdf'
data = extract_images_and_texts(pdf_path)
