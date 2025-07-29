import os
import json
import time
import sys
import pickle
from tqdm import tqdm

# ✅ Add root directory to import path BEFORE using other project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from models.text_embedding_small import get_embedding


CHUNK_DIR = "chunks"
PCA_PATH = "utils/pca_1536_to_1024.pkl"

# ✅ Load trained PCA model
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
    return chunks

def index_chunks(file_path):
    if "_150.jsonl" in file_path:
        namespace = "150"
    elif "_300.jsonl" in file_path:
        namespace = "300"
    else:
        print(f"❌ Skipping file (unknown chunk type): {file_path}")
        return

    print(f"\n📥 Indexing {file_path} into namespace: {namespace}")
    chunks = load_chunks(file_path)
    batch = []

    for chunk in tqdm(chunks, desc=f"🔗 Embedding [{namespace}]"):
        full_vector = get_embedding(chunk["text"])
        if not full_vector or len(full_vector) != 1536:
            continue

        # ➡️ Apply PCA to reduce dimensionality
        reduced_vector = pca.transform([full_vector])[0]  # → shape: (1024,)

        # ⬇️ Store the full metadata including text
        metadata = {
            **chunk["metadata"],
            "text": chunk["text"]
        }

        record = {
            "id": chunk["id"],
            "values": reduced_vector.tolist(),
            "metadata": metadata
        }
        batch.append(record)

        if len(batch) == 50:
            index.upsert(vectors=batch, namespace=namespace)
            batch = []
            time.sleep(0.2)

    if batch:
        index.upsert(vectors=batch, namespace=namespace)

    print(f"✅ Indexed {len(chunks)} chunks into namespace '{namespace}'")

def run():
    for fname in os.listdir(CHUNK_DIR):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(CHUNK_DIR, fname)
            index_chunks(fpath)

if __name__ == "__main__":
    run()
    print("🔍 Finished chunk indexing.")
