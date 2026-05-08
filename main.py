from dotenv import load_dotenv
import os

# --------------------------------------------------
# LOAD ENV VARIABLES
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# FASTAPI IMPORTS
# --------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# LLAMAINDEX IMPORTS
# --------------------------------------------------
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

# --------------------------------------------------
# LOCAL EMBEDDING MODEL
# --------------------------------------------------
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --------------------------------------------------
# MISTRAL LLM
# --------------------------------------------------
from llama_index.llms.mistralai import MistralAI

# --------------------------------------------------
# DEBUG API KEY
# --------------------------------------------------
print("MISTRAL API KEY LOADED:", os.getenv("MISTRAL_API_KEY"))

# --------------------------------------------------
# SET LOCAL EMBEDDINGS
# --------------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# --------------------------------------------------
# SET MISTRAL MODEL
# --------------------------------------------------
Settings.llm = MistralAI(
    model="mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------
documents = SimpleDirectoryReader("data").load_data()

# --------------------------------------------------
# OPTIONAL DOCUMENT DEBUG
# --------------------------------------------------
for doc in documents:
    print("\n================ DOCUMENT CONTENT ================\n")
    print(doc.text[:1000])

# --------------------------------------------------
# CREATE VECTOR INDEX
# --------------------------------------------------
index = VectorStoreIndex.from_documents(documents)

# --------------------------------------------------
# CREATE QUERY ENGINE
# --------------------------------------------------
query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    similarity_top_k=5
)

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="Holiday RAG API",
    description="Holiday Calendar Question Answering using LlamaIndex + Mistral",
    version="1.0.0"
)

# --------------------------------------------------
# ENABLE CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------
class QueryRequest(BaseModel):
    question: str

# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Holiday RAG API Running Successfully"
    }

# --------------------------------------------------
# QUERY ENDPOINT
# --------------------------------------------------
@app.post("/query")
async def query_docs(request: QueryRequest):

    try:
        # Query the RAG pipeline
        response = query_engine.query(request.question)

        return {
            "status": "success",
            "question": request.question,
            "answer": str(response)
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }