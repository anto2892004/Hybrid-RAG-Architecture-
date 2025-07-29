import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gpt_4_1_nano_judge import judge_answers

question = "What did Apple say about revenue?"
ground_truth = "Apple reported a 7% year-over-year increase in revenue due to strong iPhone sales."

answers = {
    "gemini": "Apple said revenue rose 7% thanks to iPhone sales.",
    "nova": "Revenue was flat compared to last year.",
    "llama": "Apple mentioned a 7% revenue growth driven by iPhones."
}

print("🧪 Testing GPT-4.1-nano Judge...")
result = judge_answers(question, ground_truth, answers)
print("✅ Judgment:\n", result["judgment"])
