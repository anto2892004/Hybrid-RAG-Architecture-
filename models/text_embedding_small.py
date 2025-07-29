import openai
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import NEXUS_API_KEY, NEXUS_BASE_URL, EMBEDDING_MODEL
from sklearn.decomposition import PCA
import numpy as np

# 🔌 Connect to Nexus (OpenAI-compatible API)
client = openai.OpenAI(
    api_key=NEXUS_API_KEY,
    base_url=NEXUS_BASE_URL
)

# ⚙️ Setup PCA reducer from 1536 → 1024
pca = PCA(n_components=1024)


def get_embedding(text: str) -> list:
    if not text.strip():
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    embedding = response.data[0].embedding
    return embedding  
