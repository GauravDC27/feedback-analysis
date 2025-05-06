# rag_v2/vector_store.py

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import faiss
import numpy as np
import pickle



INDEX_PATH = "dataset/faiss_index.bin"
METADATA_PATH = "dataset/metadata.pkl"

def get_embedding(text: str) -> list:
    response = client.embeddings.create(input=[text],
    model="text-embedding-ada-002")
    return response.data[0].embedding

def load_vector_store():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def query_vector_store(question: str, top_k: int = 5):
    index, metadata = load_vector_store()
    query_vector = np.array([get_embedding(question)], dtype="float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

