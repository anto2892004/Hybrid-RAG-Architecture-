import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.llama3_8b_8192 import generate_answer

prompt = "Summarize Apple's Q2 2023 earnings in one sentence."

print("🧪 Testing LLaMA 3...")
response = generate_answer(prompt)
print("✅ LLaMA 3 Response:\n", response)
