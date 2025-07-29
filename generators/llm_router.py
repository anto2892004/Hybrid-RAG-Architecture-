

import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NEXUS_API_KEY, NEXUS_BASE_URL  
from openai import OpenAI

client = OpenAI(api_key=NEXUS_API_KEY, base_url=NEXUS_BASE_URL)

GENERATOR_MODELS = {
    "gemini": "gemini-2.5-flash",
    "nova": "nova-micro",
    "llama": "llama3-8b-8192"
}

def build_prompt(query: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return (
        f"You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

def generate_with_all_models(query: str, contexts: list[str]) -> dict:
    prompt = build_prompt(query, contexts)
    outputs = {}

    for label, model in GENERATOR_MODELS.items():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            outputs[label] = response.choices[0].message.content.strip()
        except Exception as e:
            outputs[label] = f"[ERROR] {str(e)}"

    return outputs
