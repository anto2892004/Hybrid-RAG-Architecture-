import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NEXUS_API_KEY, NEXUS_BASE_URL, GENERATOR_MODELS
import openai

client = openai.OpenAI(api_key=NEXUS_API_KEY, base_url=NEXUS_BASE_URL)

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GENERATOR_MODELS["nova"],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
