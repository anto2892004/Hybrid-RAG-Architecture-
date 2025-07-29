import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NEXUS_API_KEY, NEXUS_BASE_URL, JUDGE_MODEL
import openai

# Initialize Nexus client
client = openai.OpenAI(
    api_key=NEXUS_API_KEY,
    base_url=NEXUS_BASE_URL
)

def judge_answers(question: str, ground_truth: str, answers: dict) -> dict:
    """
    Compares multiple LLM answers to a ground truth using GPT-4.1-nano as judge.
    Args:
        question (str): The original user question
        ground_truth (str): The context or chunk from vector DB
        answers (dict): { "gemini": "...", "nova": "...", "llama": "..." }
    Returns:
        dict: Scores or evaluation comments per model
    """

    model_inputs = [
        f"Question: {question}",
        f"Reference Answer (from DB): {ground_truth}",
    ]

    for name, ans in answers.items():
        model_inputs.append(f"{name} Answer: {ans}")

    prompt = "\n\n".join(model_inputs) + \
        "\n\nJudge each model on correctness, completeness, and faithfulness to the reference. Provide scores out of 10 for each.\n"

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"judgment": response.choices[0].message.content.strip()}
