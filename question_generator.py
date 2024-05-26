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
        limit= 2
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

Specific Questions Mentioning Listed Car Models: If a question directly mentions any of the car models—Tata Nexon, Tata Punch, Hyundai Exter, or Hyundai Next Gen Verna—respond with: "No further clarifications needed."

General Questions: For questions that apply universally to all cars , which would generate the same response for any car, output the specific line: "This is a general query applicable to all cars." Do not return any other output or ask for further clarification.

Specific Questions About Car Models: When a question indicates that the answer might vary among different models, without specifying a car, prompt the user for more information. Specifically, ask: "Could you please specify which car model you are referring to? Is it Tata Nexon, Tata Punch, Hyundai Exter, or Hyundai Next Gen Verna?"

If the user specifies TATA or Hyundai, ask: "Could you please specify the model? Is it Tata Nexon, Tata Punch or Hyundai Exter, Hyundai Next Gen Verna based on what he has mentioned in the query.

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

from collections import defaultdict, Counter

def get_context(documents, query):
    # Replace car names in the query
    print(f"Final query is: {replace_car_names(query)}")
    query = replace_car_names(query)

    # Document to collection mapping
    mapping = {
        'Tata Nexon': 'text_collection_nexon-owner-manual-2022',
        'Tata Punch': 'text_collection_punch-bsvi-09-09-21',
        'Hyundai Exter': 'text_collection_exter',
        'Hyundai Next Gen Verna': 'text_collection_Next_Gen_Verna'
    }
    
    # Prepare the containers for responses and contexts
    resp = []
    combined_context = {doc: {'score': [], 'content': [], 'pdf_name': [], 'page': []} for doc in documents}
    image_context = {doc: {'score': [], 'img_path': [], 'pdf_name': [], 'page': []} for doc in documents}

    # Query each document collection
    for doc in documents:
        print(f"{mapping[doc]} is this")
        resp.append(client.search(collection_name=mapping[doc], query_vector=text_embed_model.encode(query), limit=10))

    # Collect the search results and image data
    for i, r in enumerate(resp):
        doc = documents[i]
        for result in r:
            combined_context[doc]['score'].append(result.score)
            combined_context[doc]['content'].append(result.payload['content'])
            combined_context[doc]['pdf_name'].append(result.payload['pdf_name'])
            combined_context[doc]['page'].append(result.payload['page'])

            # Retrieve associated images
            image_response = retrieve_images(client, result.payload['pdf_name'], result.payload['page'], query)
            if image_response:
                for img_resp in image_response:
                    print(img_resp.payload['content'])
                    image_context[doc]['score'].append(img_resp.score)
                    image_context[doc]['img_path'].append(img_resp.payload['content'])
                    image_context[doc]['pdf_name'].append(result.payload['pdf_name'])
                    image_context[doc]['page'].append(result.payload['page'])

    # Process all texts to find top 3 ensuring at least two are from the same collection
    all_texts = []
    for doc in documents:
        for idx in range(len(combined_context[doc]['score'])):
            entry = {
                'score': combined_context[doc]['score'][idx],
                'content': combined_context[doc]['content'][idx],
                'pdf_name': combined_context[doc]['pdf_name'][idx],
                'page': combined_context[doc]['page'][idx],
                'doc': doc
            }
            all_texts.append(entry)

    # Sort by score
    all_texts_sorted = sorted(all_texts, key=lambda x: x['score'], reverse=True)

    # Select the top texts ensuring at least two are from the same document
    top_texts = []
    doc_count = Counter()
    for text in all_texts_sorted:
        top_texts.append(text)
        doc_count[text['doc']] += 1
        if len(top_texts) >= 3 and any(count >= 2 for count in doc_count.values()):
            break

    final_text_content = []
    final_image_paths = []

    # Find associated images for selected texts
    for text in top_texts:
        associated_images = [(image_context[text['doc']]['score'][i], image_context[text['doc']]['img_path'][i])
                             for i in range(len(image_context[text['doc']]['pdf_name']))
                             if image_context[text['doc']]['pdf_name'][i] == text['pdf_name'] and image_context[text['doc']]['page'][i] == text['page']]
        if associated_images:
            associated_images.sort(reverse=True, key=lambda x: x[0])
            highest_image_path = associated_images[0][1]
        else:
            highest_image_path = None

        final_text_content.append(text['content'])
        final_image_paths.append(highest_image_path)

    combined_text_content = "\n".join(final_text_content)
    print(combined_text_content)
    print(final_image_paths)
    
    return combined_text_content, final_image_paths
    # for doc in documents:
    #     print(f"{mapping[doc]} is this")
    #     resp.append(client.search(collection_name=f'{mapping[doc]}', query_vector= text_embed_model.encode(query) , limit=2))
    
    # combined_context = {
    #     'Tata Nexon': {
    #         'score': [],
    #         'content': []
    #     },
    #     'Tata Punch': {
    #         'score': [],
    #         'content': []
    #     },
    #     'Hyundai Exter': {
    #         'score': [],
    #         'content': []
    #     },
    #     'Hyundai Next Gen Verna': {
    #         'score': [],
    #         'content': []
    #     }
    # }

    # image_context = {
    #     'Tata Nexon': {
    #         'score' : [], 
    #         'img_path': []
    #     },
    #     'Tata Punch': {
    #         'score' : [], 
    #         'img_path': []
    #     },
    #     'Hyundai Exter': {
    #         'score' : [], 
    #         'img_path': []
    #     },
    #     'Hyundai Next Gen Verna': {
    #         'score' : [], 
    #         'img_path': []
    #     }
    # }
    # # for r in resp:
    # #     contents = [r.payload['content'] for r in r]
    # #     combined_context = "\n".join(contents)
    # #     for r in r:
    # #         print('--------------------------------------------------------')
    # #         print(r.score)
    # #         print(r.payload['content'])
    # #         print(r.payload['page'])
    # #         print(r.payload['pdf_name'])

    # #         pdf_name = r.payload['pdf_name']
    # #         page_number = r.payload['page']
    # # # print(pdf_name, page_number)
    # #         image_response = retrieve_images(client, pdf_name, page_number, query)
    # #         if image_response:
    # #             for resp in image_response:
    # #                 img_path = resp.payload['content']
    # #                 print(img_path)
    # #                 image_context[doc]['score'].append(resp.score)
    # #                 image_context[doc]['img_path'].append(img_path)
    #     #     print(img_path)
    #     # combined_context = " ".join([doc.page_content for doc in contexts])
    

   
        
    # for result in resp:
    #     combined_context[doc]['score'].append(result.score)
    #     combined_context[doc]['content'].append(result.payload['content'])
            
    #     pdf_name = result.payload['pdf_name']
    #     page_number = result.payload['page']
            
    #     if doc not in text_scores:
    #         text_scores[doc] = []

    #     text_scores[doc].append((result.score, pdf_name, page_number))
            
    #     image_response = retrieve_images(client, pdf_name, page_number, query)
    #     if image_response:
    #         for img_resp in image_response:
    #             img_path = img_resp.payload['content']
    #             image_context[doc]['score'].append(img_resp.score)
    #             image_context[doc]['img_path'].append(img_path)
    
    # return combined_context , image_context


def replace_car_names(query):
    car_variants = [
        "Tata Nexon",
        "Tata Punch",
        "Hyundai Exter",
        "Hyundai Next Gen Verna"
    ]

    for car in car_variants:
        query = query.replace(car, "car")
    
    query = query.replace("tata", "").replace("hyundai", "") 
    query = ' '.join(query.split())

    return query

def gen_ans(context, query):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If the user query is not in context, just say that you don't know. \
    Please do not provide any information that is not in the context. \
    Keep in mind, you will lose the job, if you answer out of CONTEXT questions.\
    CONTEXT: {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    print(f"final query is {replace_car_names(query)}")
    prompt = qa_prompt.format(
        chat_history=chat_history,
        input=replace_car_names(query),
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
    fresh_query, not_needed = retrieve_documents(retriever, llm, query)
    print(fresh_query)
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
    print("response is .... \n")
    print(str_response)
    # if len(chat_history) > 4:
    #     chat_history = chat_history[-4:]
    if ("This is a general query applicable to all cars" in str_response):

        documents =['Tata Nexon', 'Tata Punch', 'Hyundai Exter', 'Hyundai Next Gen Verna']
        context, img_context = get_context(documents, fresh_query)
        print('generating final answer....')
        response = gen_ans(context , fresh_query)
    
    elif ( "No further clarifications needed" in str_response):

        print("generating final answer....")
        context, img_context = get_context(documents, fresh_query)
        # print(context)
        response = gen_ans(context , fresh_query)
    
    else:

        response = str_response

    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    if img_context:
        if img_context[0]:
            print(img_context[0])
            return response, img_context[0]

    return response , None



if __name__ == "__main__":
    while (True):
        query = input("Enter the query: ")
        if (query == "Exit"):
            break
        print(generate_response(query)[0])
            

