import gradio as gr

def yes_man(message, history):
    print(history)
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"

gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height= 800),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Yes Man",
    description="Ask Yes Man any question",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn= None,
    clear_btn=None
).launch()