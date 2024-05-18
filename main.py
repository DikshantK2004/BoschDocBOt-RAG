from response_generator import generate_response
from langchain_community.vectorstores import FAISS
import gradio as gr



# gr.ChatInterface(random_response).launch()

def handle_query(query, history):
    return generate_response(query)


gr.ChatInterface(
    handle_query,
    chatbot=gr.Chatbot(height= 650),
    textbox=gr.Textbox(placeholder="Query", container=False, scale=7),
    title="Document QA bot",
    description="Ask any question on the TAT doc",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn= None,
    clear_btn=None
).launch(share = True)

    
# print("\nAI response:\n", response)
