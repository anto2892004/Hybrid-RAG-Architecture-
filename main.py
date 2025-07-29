
from retrievers.hybrid_retriever import hybrid_search
from generators.llm_router import generate_with_all_models
from evaluators.judge_answers import evaluate_responses  

if __name__ == "__main__":
    query = input("🔍 Enter your query: ")

    # Step 1: Hybrid Retrieval
    top_chunks = hybrid_search(query, top_k=3)

    print("\n📚 Top Retrieved Chunks:")
    top_contexts = []
    for i, match in enumerate(top_chunks, 1):
        metadata = match["metadata"]
        text = metadata.get("text", "")
        top_contexts.append(text)

        print(f"\n--- {i} ---")
        print(f"Score     : {match['score']:.4f}")
        print(f"Chunk ID  : {match['id']}")
        print(f"Source    : {metadata.get('source')}")
        print(f"Company   : {metadata.get('company')}")
        print(f"Date      : {metadata.get('date')}")
        print(f"Content   : {text[:300]}...")  

    # Step 2: Multi-Model Generation
    responses = generate_with_all_models(query, top_contexts)

    print("\n🤖 Generated Answers:")
    for model, answer in responses.items():
        print(f"\n🔹 Model: {model}\n{answer}\n")

    # Step 3: Judge Evaluation
    print("🧠 Evaluating with Judge Model (gpt-4.1-nano)...")
    judgement = evaluate_responses(query, top_contexts, responses)

    print("\n📊 Evaluation Result:")
    print(judgement)
