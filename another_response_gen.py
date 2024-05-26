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


chat_history = []
# db = FAISS.load_local('./faiss_index', get_embeddingmodel()  , allow_dangerous_deserialization= True)
# retriever = initialize_retriever(db)

pdfs = ["nexon", "verna", "punch", "exter"]
from qdrant_client import QdrantClient , models

client = QdrantClient(path = "qdrant_data_fresh_5",
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


def detect_probing_need(llm, context, user_query):
    # System prompt for detecting if probing is needed
    detect_probing_system_prompt = """You are an assistant to detect whether probing is required by AI assistant to answer the question based on given context. Use the following pieces of \
retrieved context to determine if additional clarification is needed from the user. If you feel relevant information \
is present in multiple documents, ask the user to specify which document they want the answer from. Do NOT answer the question, \
just indicate whether probing is needed and specify the area for clarification.
Example: query: tell about vehicle petrol consumption. context:- Tata Nexon: Vehicle consumes 50L at 30 kmph , Tata Punch: Vehicle consumes 50L at 40 kmph and 80L at 60 kmph.
So, response must be:- 
Yes, Could you kindly confirm what vehicle do you want the info for and for what speeds?
You will LOSE THE JOB IF YOU DON'T FOLLOW INSTRUCTIONS.
CONTEXT: {context}"""

    # Creating a chat prompt template for detecting if probing is needed
    detect_probing_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", detect_probing_system_prompt),
            ("human", "{input}"),
        ]
    )

    # Formatting the prompt with the chat history and the user query
    prompt_text = detect_probing_prompt.format(
        chat_history=chat_history,
        input=user_query,
        context = context
    )

    # Calling the LLM to generate a response
    response = llm.generate(prompts = [prompt_text])
    res = StrOutputParser().parse(response).generations[0][0].text
    return res

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




def retrieve_documents( llm, query):
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

    response = llm.generate(prompts=[prompt_text])
    fresh_query = StrOutputParser().parse(response).generations[0][0].text
    print("hi")
    # contexts = retriever.invoke(fresh_query)
    rs = []
    for pdf in pdfs:
        resp = client.search(collection_name=f'text_collection_{pdf}', query_vector= text_embed_model.encode(query) , limit=2)
        for r in resp:
            if r.score > 0.38:
                rs.append(r)
        
            
        
    contents = [r.payload['content'] for r in rs]
    combined_context = "\n\n".join(contents)

    # combined_context = " ".join([doc.page_content for doc in contexts])
    return combined_context , contents, fresh_query

def retrieve_final_query(context, query, resp):
    qa_system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
    If the user query is not in context, just say that you don't know. \
    You will lose the job, if you answer out of CONTEXT questions.\
    Keep the response in 7-8 lines
    CONTEXT: {context}"""
    
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    prompt = qa_prompt.format(
        chat_history=chat_history,
        input=query,
        context = context
    )
    # print(prompt)
    return prompt
    
def generate_response(query):
    global retriever
    global llm
    global chat_history

    retrieved_context , resp, fresh_query= retrieve_documents( llm, query)
    # print(retrieved_context)
    # pdf_name = resp[0].payload['pdf_name']
    # page_number = resp[0].payload['page']
    # print(pdf_name, page_number)
    # image_response = retrieve_images(client, pdf_name, page_number, query)
    # try:
    #     img_path = image_response[0].payload['content']
    #     print(img_path)
    # except Exception:
    #     pass
    rfinal_query = retrieve_final_query(retrieved_context, fresh_query, resp)
    # print(rfinal_query)
    
    
    res : str = detect_probing_need(llm, retrieved_context, fresh_query)
    if  res.split(',')[0].lower().strip().find("yes") != -1:
        chat_history.extend([HumanMessage(rfinal_query), AIMessage(content=res[4:])])
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]
        return res[4:]
    
    # print("\nGenerating the response....")
    
    response = llm.generate(prompts=[rfinal_query])
    parsed_response = StrOutputParser().parse(response)
    str_response = parsed_response.generations[0][0].text
    chat_history.extend([HumanMessage(content=rfinal_query), AIMessage(content=str_response)])
    
    if len(chat_history) > 4:
        chat_history = chat_history[-4:]
    # return str_response

if __name__ == "__main__":
    while True:
        query = input()
        print("hi")
        print(generate_response(query))

