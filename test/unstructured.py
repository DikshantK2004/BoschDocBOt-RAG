from unstructured.partition.pdf import partition_pdf

# elements = partition_pdf("tata.pdf")

elements_fast = partition_pdf("/home/dikshant/BOSCH/Round1/tata.pdf",
    chunking_strategy="by_title",
    strategy="fast",
    max_characters=1500,
    overlap=300,
    overlap_all= True
  )

print(elements_fast)
