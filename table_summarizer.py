from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
import pytesseract
from PIL import Image
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_community.llms.huggingface_pipeline import HuggingFaceLLM

# Function to perform OCR on an image and extract text
# def extract_text_from_image(image_path):
#     image = Image.open(image_path)
#     text = pytesseract.image_to_string(image)
#     return text

# image_path = "./table.png" 

# text = extract_text_from_image(image_path)
# print(text)

csv = 'first.csv'
# get all text of csv file
text = ""
with open(csv, 'r') as file:
    text = file.read()

llm = Ollama(model="phi3", callback_manager=CallbackManager([]))


prompt = """
You are a table summarizer bot. I have extracted data of a table into csv You are supposed to summarize each row into 1 line so that it looks neat.
    Understand the context by seeing the heading of the table and summarize each row into 1 line.
    Create output only on the table provided. Adding anything of your own means losing job.
    Complete them as sentences. Don't write any notes and all."""

pr = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "{input}"),
        ]
    )


fin = pr.format(
    input = text
)



from langchain.schema import StrOutputParser
response = llm.generate(prompts=[fin])
ans = StrOutputParser().parse(response).generations[0][0].text

print(ans)