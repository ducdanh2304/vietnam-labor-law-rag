# ⚖️ Vietnam Labor Law RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about **Vietnam's 2019 Labor Code (Bộ luật Lao động 2019)**, built with a hybrid retrieval pipeline, query rewriting, semantic routing, and a self-correcting retrieval loop (Self-RAG).

> Built as a personal project to explore production-grade RAG architecture patterns beyond naive "embed and retrieve" pipelines.

---

## 🧠 Architecture

Instead of a single embed → retrieve → generate step, the pipeline adds three layers of reasoning to improve accuracy on legal text, where precision (exact article numbers, exact wording) matters far more than in general Q&A:

```
User Question
     │
     ▼
┌─────────────────┐
│  Query Rewriter  │  → rewrites casual/vague questions into precise legal queries
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Semantic Router │  → classifies: LEGAL / OUT_OF_SCOPE / UNCLEAR
└────────┬─────────┘
         │ LEGAL
         ▼
┌─────────────────────────────┐
│   Hybrid Retriever           │
│   (BM25 + Semantic Search)   │  → ensemble of keyword + embedding search
└────────┬─────────────────────┘
         ▼
┌─────────────────┐
│    Self-RAG      │  → checks if retrieved chunks are SUFFICIENT
│  relevance check  │     if not, rewrites query and retries retrieval
└────────┬─────────┘
         ▼
┌─────────────────┐
│   LLM Answer      │  → generates answer grounded in retrieved articles,
│   (Groq / Llama)  │     always citing the specific "Điều" (Article)
└──────────────────┘
```

**Why this matters:** a plain RAG setup often retrieves loosely-related chunks and answers confidently anyway. This pipeline explicitly checks retrieval quality (Self-RAG) and rejects out-of-scope questions (Router) before generating an answer, reducing hallucinated legal citations.

---

## ✨ Features

| Component | Purpose |
|---|---|
| **Query Rewriting** | Converts colloquial questions ("bị đuổi việc thì sao") into precise legal queries |
| **Semantic Router** | Filters out-of-scope or unclear questions before wasting a retrieval call |
| **Hybrid Retrieval** | Combines BM25 (keyword) + vector search (semantic) via `EnsembleRetriever` for better recall on legal terminology |
| **Self-RAG** | LLM judges whether retrieved chunks are sufficient; retries with a reformulated query if not |
| **Automated Evaluation** | LLM-as-judge scoring harness against a hand-curated test set with ground-truth article citations |
| **Streamlit UI** | Chat interface with conversation memory and a "show optimized query" debug view |

---

## 🛠️ Tech Stack

- **LLM**: Groq (Llama 3.1 8B / Llama 3.3 70B)
- **Framework**: LangChain
- **Vector Store**: ChromaDB
- **Embeddings**: `keepitreal/vietnamese-sbert` (HuggingFace)
- **Keyword Search**: BM25
- **UI**: Streamlit

---

## 📁 Project Structure

```
vietnam-labor-law-rag/
├── data/
│   └── luat_lao_dong.txt       # source text of the 2019 Labor Code
├── src/
│   ├── ingest.py                # builds/rebuilds the Chroma vector store
│   ├── retriever.py              # hybrid retriever (BM25 + semantic)
│   ├── query_rewriter.py         # LLM-based query rewriting
│   ├── router.py                 # LEGAL / OUT_OF_SCOPE / UNCLEAR classification
│   ├── self_rag.py               # retrieval sufficiency check
│   ├── chain.py                  # builds the end-to-end answer chain
│   └── ui.py                     # Streamlit chat interface
├── scripts/
│   └── run_ingest.py             # entry point to (re)build the vector store
├── evaluation/
│   └── evaluate.py               # LLM-as-judge scoring against test cases
├── requirements.txt
└── .env.example
```

---

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/vietnam-labor-law-rag.git
cd vietnam-labor-law-rag
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Add your [Groq API key](https://console.groq.com) to `.env`:

```
GROQ_API_KEY=your_key_here
```

### 3. Build the vector store

```bash
python scripts/run_ingest.py
```

This reads `data/luat_lao_dong.txt`, chunks it (splitting on `Điều` / `Khoản` boundaries to preserve legal structure), embeds it, and persists it to `chroma_db/`.

### 4. Run the chatbot

```bash
streamlit run src/ui.py
```

### 5. (Optional) Run the evaluation suite

```bash
python -m evaluation.evaluate
```

Scores each answer 1–5 using an LLM judge against ground-truth article citations, and prints an average score.

---

## 📊 Evaluation

A 10-question test set covering both easy (direct retrieval) and harder (multi-hop / less obvious phrasing) legal questions is scored automatically using an LLM-as-judge rubric (1 = wrong, 5 = correct with exact article citation).

Current baseline: **3.3 / 5**

This is an early baseline — see [Roadmap](#-roadmap) for planned improvements.

---

## 🗺️ Roadmap

- [ ] Improve chunking strategy for articles with nested `Khoản` (sub-clauses)
- [ ] Add re-ranking step after hybrid retrieval
- [ ] Expand evaluation set beyond 10 questions
- [ ] Add source highlighting in the UI (show exact retrieved passage)
- [ ] Deploy a live demo (Streamlit Community Cloud)

---

## ⚠️ Disclaimer

This project is for educational/demonstration purposes only and does not constitute legal advice. Always consult a qualified legal professional for actual labor law matters.

---

## 📄 License

MIT
