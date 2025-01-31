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
db = FAISS.load_local('./faiss_index', get_embeddingmodel()  , allow_dangerous_deserialization= True)
retriever = initialize_retriever(db)

llm = Ollama(model="phi3", callback_manager=CallbackManager([]))

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

    response = llm.generate(prompts=[prompt_text])
    fresh_query = StrOutputParser().parse(response).generations[0][0].text

    contexts = retriever.invoke(fresh_query)
    combined_context = " ".join([doc.page_content for doc in contexts])
    return combined_context

def retrieve_final_query(context, query):
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If the user query is not in context, just say that you don't know. \
    Please do not provide any information that is not in the context. \
    Keep in mind, you will lose the job, if you answer out of CONTEXT questions.\
    Keep the response in 7-8 lines
    {context}
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
        context=context
    )
    return prompt
    
def generate_response(query):
    global retriever
    global llm
    global chat_history

    retrieved_context= retrieve_documents(retriever, llm, query)
    print(retrieved_context)
    rfinal_query = retrieve_final_query(retrieved_context, query)
    print(rfinal_query)
    
    
    print("\nGenerating the response....")
    
    response = llm.generate(prompts=[rfinal_query])
    parsed_response = StrOutputParser().parse(response)
    str_response = parsed_response.generations[0][0].text
    chat_history.extend([HumanMessage(content=query), AIMessage(content=str_response)])
    
    if len(chat_history) > 4:
        chat_history = chat_history[-4:]
    return str_response

# if __name__ == "__main__":
#     query = "How to drive on Gradients?"
#     print(generate_response(query))

