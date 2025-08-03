
---

# 📊 Multi-Generator RAG with Hybrid Retrieval and Judge Evaluation

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that enhances response accuracy and diversity using **multiple LLMs** and a **judging model**, built around hybrid retrieval of PCA-reduced vector chunks from company earnings call transcripts.

---

## 🔧 Tech Stack

* **Language**: Python 3.10+
* **LLM API**: OpenAI-compatible models via Nexus API
* **Embedding Model**: `text-embedding-3-small`
* **Vector DB**: Pinecone (1024D embeddings, with namespaces)
* **Dimensionality Reduction**: PCA (1536 → 1024)
* **LLMs Used for Generation**:

  * `gemini-2.5-flash`
  * `nova-micro`
  * `llama3-8b-8192`
* **Judge Model**: `gpt-4.1-nano`

---

## ⚙️ Project Pipeline

```text
User submits query
        │
        ▼
Hybrid Retrieval (PCA-reduced embeddings) from Pinecone
        │
        ▼
Top-K relevant chunks selected from 2 namespaces (150, 300 token sizes)
        │
        ▼
Each LLM receives the same query + top chunks and generates a response
        │
        ▼
All responses are evaluated by the judge model based on 4 metrics:
  • Relevance  • Factual Accuracy  • Completeness  • Clarity
        │
        ▼
Best answer is selected and all evaluations are displayed
```

---

## 📦 Features

* ✅ **Hybrid Retriever**: Uses both 150-token and 300-token chunk namespaces.
* ✅ **PCA Optimization**: Reduces embedding size for faster, more efficient vector search.
* ✅ **Multi-LLM Generator**: Leverages the strengths of different LLMs to diversify answers.
* ✅ **Judge Evaluation**: Fair evaluation by GPT-4.1-nano using defined scoring metrics.
* ✅ **Modular Design**: Easily pluggable components for retrieval, generation, and judging.

---

## 🧠 Evaluation Metrics

Each LLM response is evaluated on a scale of **1 to 10** for:

* **Relevance**: How well the answer addresses the query.
* **Factual Accuracy**: Correctness of information based on the context chunks.
* **Completeness**: Whether all parts of the question are answered.
* **Clarity**: Coherence and readability of the explanation.

The best response is selected with a justification.

---

## 📁 Directory Structure

```
RAG-project/
│
├── main.py                        # End-to-end pipeline
├── config.py                      # API keys and config
│
├── retrievers/
│   └── hybrid_retriever.py       # PCA + Pinecone + hybrid namespace search
│
├── generators/
│   └── llm_router.py             # Multi-model LLM generation logic
│
├── evaluators/
│   └── judge_answers.py          # Judge model to evaluate responses
│
├── models/
│   └── text_embedding_small.py   # Embedding via Nexus
│
├── utils/
│   └── pca_1536_to_1024.pkl      # Precomputed PCA model for dim reduction
```

---

## 🗃️ Data & Chunking

* **Source**: Earnings call transcripts from top tech companies.
* **Chunk Sizes**: 150 and 300 tokens.
* **Overlap**: 30 and 50 tokens respectively.
* **Metadata stored**:

  * `company`, `date`, `source`, `chunk_index`, `chunk_size`, `text`

---

## 🚀 How to Run

```bash
# Set up virtual environment and install dependencies
pip install -r requirements.txt

# Add your config in config.py (API keys, index name)

# Run the pipeline
python main.py
```

---

## 🌐 Future Improvements

* [ ] Streamlit-based UI for interactive use
* [ ] Caching of judge responses
* [ ] Support for user-defined prompt templates
* [ ] Asynchronous generation for faster response

---

## 📝 License

MIT License. You are free to use, modify, and share this project with attribution.

---
