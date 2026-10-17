# RAG-Project: Ask Questions About Your Own PDF Notes

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your own PDF documents and get answers grounded in their content. I built it around my Operating Systems course notes, but it works with any set of PDFs.

Built from scratch in plain Python, so every step (loading, chunking, embedding, retrieval, generation) is visible and easy to debug.

## How it works

```
PDFs ──► Load pages ──► Split into chunks ──► Embed chunks ──► Store in ChromaDB
                                                                     │
Question ──► Embed question ──► Retrieve top-k similar chunks ◄──────┘
                                         │
                                         ▼
                        Groq LLM answers using only those chunks
```

1. **Load:** `PyMuPDFLoader` reads every PDF in `data/pdf/` and keeps the file name and page number as metadata.
2. **Chunk:** `RecursiveCharacterTextSplitter` splits pages into 1000-character chunks with 200 characters of overlap.
3. **Embed:** `sentence-transformers` (`all-MiniLM-L6-v2`) turns each chunk into a 384-dimensional vector.
4. **Store:** Vectors and metadata are saved in a persistent ChromaDB collection.
5. **Retrieve:** The question is embedded the same way, and the most similar chunks are returned.
6. **Generate:** The retrieved chunks are passed to an LLM on Groq with an instruction to answer only from that context.

## Tech stack

| Part | Tool |
|---|---|
| Language | Python 3.12+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| PDF loading | PyMuPDF via LangChain loaders |
| Chunking | LangChain text splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB (persistent) |
| LLM | Groq API (via `langchain-groq`) |
| Environment | Jupyter notebook in VS Code |

## Project structure

```
RAG-Project/
├── notebook/
│   └── pdf_loader.ipynb    # the full pipeline, step by step
├── data/
│   ├── pdf/                # put your PDFs here (not tracked by Git)
│   └── vector_store/       # generated ChromaDB files (not tracked by Git)
├── pyproject.toml          # dependencies
├── uv.lock
└── .env                    # your API key (not tracked by Git)
```

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/Logitghost/RAG-Project.git
cd RAG-Project
```

**2. Install dependencies**
```bash
uv sync
```

**3. Add your PDFs**

Create the folder `data/pdf/` and put your PDF files in it. The PDFs are not included in this repo.

**4. Add your Groq API key**

Get a free key at [console.groq.com/keys](https://console.groq.com/keys), then create a file named `.env` in the project root:
```
GROQ_API_KEY=your_key_here
```
Never commit this file. It is already listed in `.gitignore`.

**5. Run the notebook**

Open `notebook/pdf_loader.ipynb` in VS Code, select the `.venv` kernel, and run the cells from top to bottom. The first run downloads the embedding model (about 90 MB).

**6. Ask a question**
```python
print(rag_answer("What is a process control block?", rag_retriever))
```

## What I learned

- Why chunk size and overlap matter: bad chunks lead to bad retrieval, no matter how good the LLM is.
- How embeddings and distance metrics work in a vector database, including why Chroma's default distance isn't cosine.
- How to keep the LLM grounded by restricting it to retrieved context.
- Managing secrets and environments properly (`.env`, `.gitignore`, uv).

## Limitations and next steps

- Add a retrieval evaluation set (recall@k) to measure improvements instead of guessing.
- Try hybrid search (BM25 + vectors) and a reranker.
- Deduplicate overlapping source PDFs before embedding.
- Add source citations (file and page) to the final answer.
- Wrap it in a simple UI (Streamlit or Gradio).
