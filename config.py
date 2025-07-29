# config.py

# === 🔐 Nexus API Configuration ===
NEXUS_API_KEY = "sk-Z6rAzMELcOVJO0T9FjLjZw"
NEXUS_BASE_URL = "https://apidev.navigatelabsai.com"

# === 🤖 Model Names (served via Nexus) ===
EMBEDDING_MODEL = "text-embedding-3-small"

GENERATOR_MODELS = {
    "gemini": "gemini-2.5-flash",
    "nova": "nova-micro",
    "llama": "llama3-8b-8192"
}

JUDGE_MODEL = "gpt-4.1-nano"




# === 📦 Pinecone VectorDB Configuration ===
PINECONE_API_KEY = "pcsk_2cAriR_S3w7EVp6MF3fGFzhh1X8HfjSSFi5A8h6UfnQPQCTDNNUWLyHstS4N1kfohhsH3a"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "transcripts-300"  


# === ✂️ Chunking Settings ===
CHUNK_SIZES = {
    "small": 150,
    "large": 300
}

CHUNK_OVERLAPS = {
    "small": 30,
    "large": 40
}


TOP_K_PER_CHUNK = 5           
FINAL_CONTEXT_SIZE = 3       
MAX_TOKENS_PER_PROMPT = 1024  
