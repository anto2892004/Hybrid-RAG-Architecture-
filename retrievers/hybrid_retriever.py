import os
import sys
import pickle
import numpy as np
from operator import itemgetter
from pinecone import Pinecone

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from models.text_embedding_small import get_embedding

# === Constants ===
PCA_PATH = "utils/pca_1536_to_1024.pkl"
TOP_K_EACH = 10        
TOP_K_FINAL = 6        

# === Load PCA ===
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)

# === Initialize Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def reduce_embedding_dim(embedding: list[float]) -> list[float]:
    return pca.transform([embedding])[0].tolist()  # 1536 → 1024

def search_namespace(query_vector, namespace: str, top_k: int):
    response = index.query(
        namespace=namespace,
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    return response["matches"]  # List of dicts

def hybrid_search(query: str, top_k_each=TOP_K_EACH, top_k=TOP_K_FINAL):
    """
    Hybrid search from two namespaces (150, 300) and returns top_k results combined.
    """
    # 1. Embed query
    full_embedding = get_embedding(query)
    if not full_embedding or len(full_embedding) != 1536:
        raise ValueError("Invalid embedding dimension")

    # 2. Reduce dimensions using PCA
    reduced_vector = reduce_embedding_dim(full_embedding)

    # 3. Query both namespaces
    results_150 = search_namespace(reduced_vector, "150", top_k_each)
    results_300 = search_namespace(reduced_vector, "300", top_k_each)

    # 4. Combine and sort by score (higher is better)
    combined = results_150 + results_300
    combined.sort(key=itemgetter("score"), reverse=True)

    # 5. Return top_k from combined
    return combined[:top_k]

# === Test it ===
if __name__ == "__main__":
    query = input("🔍 Enter your query: ")
    top_chunks = hybrid_search(query)

    print("\n📚 Top Retrieved Chunks:")
    for i, match in enumerate(top_chunks, 1):
        metadata = match["metadata"]
        print(f"\n--- {i} ---")
        print(f"Score     : {match['score']:.4f}")
        print(f"Chunk ID  : {match['id']}")
        print(f"Source    : {metadata.get('source')}")
        print(f"Company   : {metadata.get('company')}")
        print(f"Date      : {metadata.get('date')}")
        print(f"Content   : {metadata.get('text', '[text hidden or truncated]')[:300]}...")
