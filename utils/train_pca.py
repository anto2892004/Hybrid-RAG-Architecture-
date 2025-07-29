import os
import sys

# Add root directory (RAG-project/) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.text_embedding_small import get_embedding
import numpy as np
from sklearn.decomposition import PCA
import pickle

# === Configuration ===
NUM_SAMPLES = 1200
OUTPUT_DIM = 1024
OUTPUT_PATH = "utils/pca_1536_to_1024.pkl"

BASE_TEXTS = [
    "Apple saw record growth in Q2 earnings.",
    "Cloud services continued to drive revenue.",
    "Strong iPhone sales helped boost profit margins.",
    "MacBook sales dipped slightly this quarter.",
    "The company plans to expand in Asia markets.",
    "Supply chain issues impacted delivery timelines.",
    "Subscription services grew by 15% year-over-year.",
    "AI investments will increase next year.",
    "Advertising revenue rose sharply.",
    "R&D spending reached an all-time high.",
]
SAMPLE_TEXTS = BASE_TEXTS * (NUM_SAMPLES // len(BASE_TEXTS) + 1)

# === Generate Embeddings ===
print("🔄 Generating sample embeddings...")
vectors = []
for text in SAMPLE_TEXTS[:NUM_SAMPLES]:
    vec = get_embedding(text)
    if len(vec) == 1536 and np.any(vec):  # skip all-zero vectors
        vectors.append(vec)

print(f"✅ Collected {len(vectors)} valid embeddings.")

# === Convert to NumPy array
vectors = np.array(vectors, dtype=np.float32)

# === Train PCA
print(f"🧠 Training PCA to reduce 1536 → {OUTPUT_DIM}...")
try:
    pca = PCA(n_components=OUTPUT_DIM, svd_solver="auto")
    pca.fit(vectors)

    # Save model
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(pca, f)

    print(f"✅ PCA model saved to: {OUTPUT_PATH}")

except Exception as e:
    print("❌ PCA training failed:")
    print(e)
