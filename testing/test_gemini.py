import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gemini_2_5_flash import generate_answer

prompt = "Summarize Apple's Q2 2023 earnings in one sentence."

print("🧪 Testing Gemini...")
response = generate_answer(prompt)
print("✅ Gemini Response:\n", response)
