from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.schema import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from retriever_initializer import initialize_retriever
from embedding import get_embeddingmodel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

import re
chat_history = []
db = FAISS.load_local('./faiss_index', get_embeddingmodel()  , allow_dangerous_deserialization= True)
retriever = initialize_retriever(db)


from qdrant_client import QdrantClient , models

client = QdrantClient(path = "/Users/arushigarg/Desktop/bosch/BoschDocBOt-RAG/qdrant_yes_i_have_hope",
                      timeout= 3000)



from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np


# Text embedding model
# Image embedding model using CLIP
class CLIPEmbedding:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        # print(outputs.cpu().numpy())
        return outputs.cpu().numpy()
    
    def embed_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs.cpu().numpy()

image_embed_model = CLIPEmbedding()

text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')


llm = Ollama(model="phi3", callback_manager=CallbackManager([]))



def retrieve_images(client, pdf_name, page_number,query):
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="pdf_name",  # Assuming 'pdf_name' is the field in your collection
                match=models.MatchValue(
                    value=pdf_name  # Specific PDF name extracted from text query
                ),
            ),
            models.FieldCondition(
                key="page",  # Assuming 'page' is the field in your collection
                match=models.MatchValue(
                    value=page_number  # Specific page number extracted from text query
                ),
            )
        ]
    )
    
    image_response = client.search(
        collection_name='image_collection',
        query_vector=image_embed_model.embed_text(query)[0], 
        query_filter=query_filter,
        limit=1
    )
    
    return image_response



def process_question(question):
    # Lowercase the question for standard processing
    question = question.lower()
    
    # Dictionary mapping variants to standard car names
    car_variants = {
        "nexon": "Tata Nexon",
        "punch": "Tata Punch",
        "exter": "Hyundai Exter",
        "hyundai exter": "Hyundai Exter",
        "verna": "Hyundai Next Gen Verna",
        "hyundai verna": "Hyundai Next Gen Verna",
        "next gen verna": "Hyundai Next Gen Verna"
    }
    car_present = []
    # Replace variants with standard names and check if any standard name is mentioned
    for variant, standard in car_variants.items():
        
        pattern = r'\b' + re.escape(variant) + r'\b'
        if re.search(pattern, question ):
            question = re.sub(pattern, standard, question, flags=re.IGNORECASE)
            print(question)
            car_present.append(standard)
    return question , car_present


def retrieve_documents(retriever, llm, query):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    You will LOSE THE JOB IF YOU DON'T FOLLOW INSTRUCTIONS.
    One more thing don't use synonyms, reformulation means embedding context of whole chat into a single query"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt_text = contextualize_q_prompt.format(
        chat_history=chat_history,
        input=query
    )

    # if chat_history:
    response = llm.generate(prompts=[prompt_text])
    fresh_query = StrOutputParser().parse(response).generations[0][0].text
    fresh_query = process_question(fresh_query)
    # print(fresh_query)
    # else:
    #     fresh_query = query
    # print("hi")
    # # contexts = retriever.invoke(fresh_query)
    # resp_1 = client.search(collection_name='text_collection_exter', query_vector= text_embed_model.encode(fresh_query) , limit=2)
    # # resp_2 = client.search(collection_name='text_collection_nexon-owner-manual-2022', query_vector= text_embed_model.encode(query) , limit=2)
    # # resp_3 = client.search(collection_name='text_collection_Next_Gen_Verna', query_vector= text_embed_model.encode(query) , limit=2)
    # # resp_4 = client.search(collection_name='text_collection_punch-bsvi-09-09-21', query_vector= text_embed_model.encode(query) , limit=2)
    # # props = resp[0].__dict__.keys()
    # responses = [resp_1]
    # combined_context = ""
    # for resp in responses:
    #     contents = [r.payload['content'] for r in resp]
    #     combined_context = "\n".join(contents)
    #     for r in resp:
    #         print('--------------------------------------------------------')
    #         print(r.score)
    #         print(r.payload['content'])
    #         print(r.payload['page'])
    #         print(r.payload['pdf_name'])
    #     # combined_context = " ".join([doc.page_content for doc in contexts])
    
    # return combined_context , resp
    return fresh_query

def retrieve_final_query(query):
    qa_system_prompt = """
    You are an assistant dedicated to generating questions. Please adhere strictly to these instructions:

General Questions: For questions that apply universally to all cars , which would generate the same response for any car, output the specific line: "This is a general query applicable to all cars." Do not return any other output or ask for further clarification.

Specific Questions About Car Models: When a question indicates that the answer might vary among different models, without specifying a car, prompt the user for more information. Specifically, ask: "Could you please specify which car model you are referring to? Is it Tata Nexon, Tata Punch, Hyundai Exter, or Hyundai Next Gen Verna?"

If the user specifies TATA or Hyundai, ask: "Could you please specify the model? Is it Tata Nexon, Tata Punch or Hyundai Exter, Hyundai Next Gen Verna based on what he has mentioned in the query.

Specific Questions Mentioning Listed Car Models: If a question directly mentions any of the car models—Tata Nexon, Tata Punch, Hyundai Exter, or Hyundai Next Gen Verna—respond with: "No further clarifications needed."

Questions About Unlisted Car Models: If a question mentions a car model that is not Tata Nexon, Tata Punch, Hyundai Exter, or Hyundai Next Gen Verna, respond by saying: "I don't have information about that car model. My expertise is limited to Tata Nexon, Tata Punch, Hyundai Exter, and Hyundai Next Gen Verna."

Critical Note: Your primary role is to generate appropriate follow-up questions or provide clarifications as specified above. Do not answer the questions directly. It's crucial that you follow these instructions to ensure accurate task execution.
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt = qa_prompt.format(
        chat_history=chat_history,
        input=query,
        context= " "
    )
    return prompt

def get_context(documents, query):
    resp = []
    mapping = {
        'Tata Nexon': 'text_collection_nexon-owner-manual-2022',
        'Tata Punch': 'text_collection_punch-bsvi-09-09-21',
        'Hyundai Exter': 'text_collection_exter',
        'Hyundai Next Gen Verna': 'text_collection_Next_Gen_Verna'
    }
    for doc in documents:
        print(f"{mapping[doc]} is this")
        resp.append(client.search(collection_name=f'{mapping[doc]}', query_vector= text_embed_model.encode(query) , limit=2))
    combined_context = ""
    for r in resp:
        contents = [r.payload['content'] for r in r]
        combined_context = "\n".join(contents)
        for r in r:
            print('--------------------------------------------------------')
            print(r.score)
            print(r.payload['content'])
            print(r.payload['page'])
            print(r.payload['pdf_name'])
        # combined_context = " ".join([doc.page_content for doc in contexts])
    return combined_context

def gen_ans(context, query):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If the user query is not in context, just say that you don't know. \
    Please do not provide any information that is not in the context. \
    Keep in mind, you will lose the job, if you answer out of CONTEXT questions.\
    Keep the response in 7-8 lines
    CONTEXT: {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt = qa_prompt.format(
        chat_history=chat_history,
        input=query,
        context=context
    )

    response = llm.generate(prompts=[prompt])
    parsed_response = StrOutputParser().parse(response)
    str_response = parsed_response.generations[0][0].text
    chat_history.extend([HumanMessage(content=query), AIMessage(content=str_response)])
    return str_response


def generate_response(query):
    global retriever
    global llm
    global chat_history
    query , documents = process_question(query)
    # for doc in documents:
    #     print(doc)
    fresh_query = retrieve_documents(retriever, llm, query)
    # print(retrieved_context)
    # pdf_name = resp[0].payload['pdf_name']
    # page_number = resp[0].payload['page']
    # print(pdf_name, page_number)
    # image_response = retrieve_images(client, pdf_name, page_number, query)
    # if image_response:
    #     img_path = image_response[0].payload['content']
    #     print(img_path)
    rfinal_query = retrieve_final_query(fresh_query)
    # print(rfinal_query)
    
    
    print("\nGenerating the response....")
    
    response = llm.generate(prompts=[rfinal_query])
    parsed_response = StrOutputParser().parse(response)
    str_response = parsed_response.generations[0][0].text
    chat_history.extend([HumanMessage(content=query), AIMessage(content=str_response)])
    # print(str_response)
    # if len(chat_history) > 4:
    #     chat_history = chat_history[-4:]
    if ("This is a general query applicable to all cars " in str_response):

        documents =['Tata Nexon', 'Tata Punch', 'Hyundai Exter', 'Hyundai Next Gen Verna']
        context = get_context(documents, query)
        print('generating final answer....')
        response = gen_ans(context , query)
    
    elif ( "No further clarifications needed" in str_response):

        print("generating final answer....")
        context = get_context(documents, query)
        # print(context)
        response = gen_ans(context , query)
    
    else:

        response = str_response

    return response



if __name__ == "__main__":
    while (True):
        query = input("Enter the query: ")
        if (query == "Exit"):
            break
        print(generate_response(query))
            

