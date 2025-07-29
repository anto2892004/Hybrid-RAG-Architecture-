import streamlit as st
from retrievers.hybrid_retriever import hybrid_search
from generators.llm_router import generate_with_all_models
from evaluators.judge_answers import evaluate_responses

st.set_page_config(page_title="Multi-RAG Assistant", layout="wide")

st.title("🔍 Multi-RAG: Ask Your Question")

query = st.text_input("Enter your query:")

if st.button("Submit") and query.strip():
    with st.spinner("Retrieving relevant chunks..."):
        top_chunks = hybrid_search(query, top_k=3)
        top_contexts = [chunk["metadata"]["text"] for chunk in top_chunks]

    st.subheader("📚 Top Retrieved Chunks")
    for i, chunk in enumerate(top_chunks, 1):
        st.markdown(f"**{i}. {chunk['metadata'].get('source')}** ({chunk['score']:.4f})")
        st.code(chunk["metadata"].get("text", "")[:500] + "...")

    with st.spinner("Generating answers from all models..."):
        responses = generate_with_all_models(query, top_contexts)

    st.subheader("🤖 Model Responses")
    for model, answer in responses.items():
        st.markdown(f"**🔹 {model.upper()}**")
        st.write(answer)

    with st.spinner("Evaluating responses with Judge..."):
        evaluation = evaluate_responses(query, top_contexts, responses)

    st.subheader("🧑‍⚖️ Evaluation Summary")
    st.code(evaluation)

else:
    st.info("Enter a query above and click Submit.")
