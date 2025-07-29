import os
import json
import re
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

DATA_DIR = "data/Transcripts"  
CHUNK_DIR = "chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)


CHUNK_CONFIGS = [
    {"size": 300, "overlap": 50},
    {"size": 150, "overlap": 30}
]

def tokenize(text):
    return word_tokenize(text)

def chunk_tokens(tokens, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        start += chunk_size - overlap
    return chunks

def parse_filename(filename):
    # Extract date from file name like "2016-Apr-26-AAPL.txt"
    date_match = re.match(r"(\d{4}-[A-Za-z]{3}-\d{2})", filename)
    return date_match.group(1) if date_match else "unknown"

def process_transcript(file_path, company):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)
    base_filename = os.path.basename(file_path)
    date_str = parse_filename(base_filename)

    all_chunks = {config["size"]: [] for config in CHUNK_CONFIGS}

    for config in CHUNK_CONFIGS:
        size, overlap = config["size"], config["overlap"]
        token_chunks = chunk_tokens(tokens, size, overlap)

        for i, chunk in enumerate(token_chunks):
            chunk_text = " ".join(chunk)
            chunk_data = {
                "id": f"{company}_{date_str}_{size}_{i}",
                "text": chunk_text,
                "metadata": {
                    "company": company,
                    "date": date_str,
                    "chunk_index": i,
                    "chunk_size": size,
                    "source": base_filename
                }
            }
            all_chunks[size].append(chunk_data)

    return all_chunks

def save_chunks(company, chunks_dict):
    for size, chunks in chunks_dict.items():
        out_path = os.path.join(CHUNK_DIR, f"{company}_{size}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

def run_chunking():
    for company in os.listdir(DATA_DIR):
        company_dir = os.path.join(DATA_DIR, company)
        if not os.path.isdir(company_dir):
            continue

        print(f"\n🔍 Processing company: {company}")
        all_chunks = {config["size"]: [] for config in CHUNK_CONFIGS}

        for fname in tqdm(os.listdir(company_dir), desc=f"📄 {company}"):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(company_dir, fname)
            file_chunks = process_transcript(fpath, company)
            for size in all_chunks:
                all_chunks[size].extend(file_chunks[size])

        save_chunks(company, all_chunks)
        print(f"✅ Saved chunks for {company} (150 & 300)\n")

if __name__ == "__main__":
    run_chunking()
