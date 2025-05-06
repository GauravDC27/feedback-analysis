import pandas as pd
import os
from openai import OpenAI
from vector_store import get_embedding
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import faiss
import numpy as np
import pickle
from tqdm import tqdm


DATASET_PATH = "dataset/dataset.csv"
INDEX_PATH = "dataset/faiss_index.bin"
METADATA_PATH = "dataset/metadata.pkl"


def build_vector_store():
    df = pd.read_csv(DATASET_PATH)
    texts, metadata = [], []

    for _, row in df.iterrows():
        chunk = f"Customer feedback from {row['Location']} about {row['Product']} (Score: {row['Score']}): {row['Feedback Text']}"
        if isinstance(row["Feedback Text"], str) and row["Feedback Text"].strip():
            texts.append(chunk)
            metadata.append(dict(row))

    embeddings = [get_embedding(text) for text in tqdm(texts)]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Stored {len(embeddings)} vectors in FAISS.")

if __name__ == "__main__":
    build_vector_store()

