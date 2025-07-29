import os
import sys
from openai import OpenAI

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NEXUS_API_KEY, NEXUS_BASE_URL

client = OpenAI(api_key=NEXUS_API_KEY, base_url=NEXUS_BASE_URL)

JUDGE_MODEL = "gpt-4.1-nano"

EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator. Analyze the following responses to the question below.

Question:
{query}

Context (for reference):
{context}

Responses from different models:
{responses_block}

Evaluate each response on a scale of 1 to 10 for the following metrics:
- Relevance to the question
- Factual accuracy based on the context
- Completeness of the answer
- Clarity of the explanation

After rating all responses, choose the best one and explain why.

Output format (strictly follow):
<model>: Relevance=X, Accuracy=Y, Completeness=Z, Clarity=W
...
Best: <model>
Justification: <why that model was chosen>
"""

def build_response_block(responses: dict) -> str:
    return "\n".join([f"{model}: {answer}" for model, answer in responses.items()])

def evaluate_responses(query: str, context_chunks: list[str], responses: dict) -> dict:
    context_text = "\n---\n".join(context_chunks)
    responses_block = build_response_block(responses)
    prompt = EVAL_PROMPT_TEMPLATE.format(
        query=query,
        context=context_text,
        responses_block=responses_block
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict evaluator for AI-generated responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024
    )

    return response.choices[0].message.content.strip()
