from llama_index.multi_modal_llms.ollama import OllamaMultiModal

mm_model = OllamaMultiModal(model="phi3")


# from llama_index import SimpleDirectoryReader
from pathlib import Path
from llama_index.readers.file import UnstructuredReader
from llama_index.core.schema import ImageDocument


# import nltk
# nltk.download('averaged_perceptron_tagger')


loader = UnstructuredReader()
documents = loader.load_data(file=Path("./tesla_2021_10k.htm"))

image_doc = ImageDocument(image_path="./shanghai.jpg")


from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

print('hello')
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)



vector_index = VectorStoreIndex.from_documents(
    documents, embed_model=Settings.embed_model
)
query_engine = vector_index.as_query_engine()

from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline, FnComponent

query_prompt_str = """\
Please expand the initial statement using the provided context from the Tesla 10K report.

{initial_statement}

"""
query_prompt_tmpl = PromptTemplate(query_prompt_str)

# MM model --> query prompt --> query engine
qp = QueryPipeline(
    modules={
        "mm_model": mm_model.as_query_component(
            partial={"image_documents": [image_doc]}
        ),
        "query_prompt": query_prompt_tmpl,
        "query_engine": query_engine,
    },
    verbose=True,
)
qp.add_chain(["mm_model", "query_prompt", "query_engine"])
rag_response = qp.run("Which Tesla Factory is shown in the image?")