import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import Ollama
from response_generator import generate_response

# Initialize each model
def get_llm(model_name):
    if model_name == "phi":
        return Ollama(model="phi3")
    elif model_name == "llama2":
        return AutoModelForSeq2SeqLM.from_pretrained("meta-llama/Llama-2-7b")
    elif model_name == "deepseek":
        return Ollama(model="deepseek-7b")
    elif model_name == "distilbart":
        return AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    elif model_name == "pegasus":
        return AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    elif model_name == "bart":
        return AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    else:
        raise ValueError("Unknown model name")

import time
# Test the models with a query
def test_models(query):
    models = ["llama2", "distilbart", "pegasus", "bart"]
    results = {}
    
    for model_name in models:
        print(f"Testing {model_name}...")
        llm = None
        if model_name in ["phi", "deepseek"]:
            llm = get_llm(model_name)
            
        else:
            model = get_llm(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        
        start = time.time()
        result = generate_response(query, llm)
        end = time.time()
        results[model_name] = {
            "output": result,
            "time": end - start
        }
        # print(f"Result from {model_name}: {result}\n")
    
    return results

# Example query
query = "What precautions should pregnant women take while driving?"

# Test the models
results = test_models(query)

# Print results
for model, output in results.items():
    print(f"Model: {model}------------------------\nOutput: {output['output']}\nTime: {output['time']}\n\n")

