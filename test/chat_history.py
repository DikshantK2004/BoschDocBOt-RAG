from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Initialize chat history
chat_history = []

# Function to create the formatted prompt
def create_prompt(chat_history, question):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Format the template with the chat history and current question
    prompt_text = contextualize_q_prompt.format(
        chat_history=chat_history,
        input=question
    )
    
    return prompt_text

# Add first interaction
question = "What is Task Decomposition?"
ai_msg_1 = AIMessage(content="It is very important")
chat_history.extend([HumanMessage(content=question), ai_msg_1])

# Create and print the prompt after first interaction
formatted_prompt = create_prompt(chat_history, question)
print("Prompt after first interaction:\n", formatted_prompt)

# Add second interaction
second_question = "What are common ways of doing it?"
ai_msg_2 = AIMessage(content="Not sure what you are talking about")
chat_history.extend([HumanMessage(content=second_question), ai_msg_2])

# Create and print the prompt after second interaction
formatted_prompt = create_prompt(chat_history, second_question)
print("\nPrompt after second interaction:\n", formatted_prompt)
