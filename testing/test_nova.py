import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.nova_micro import generate_answer

prompt = "Summarize Apple's Q2 2023 earnings in one sentence."

print("🧪 Testing Nova...")
response = generate_answer(prompt)
print("✅ Nova Response:\n", response)
