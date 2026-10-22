# 02 · RAG-based Student Handbook Assistant

A mini **Retrieval-Augmented Generation (RAG)** system that answers students' questions about attendance, examinations, academic leave, library and discipline. It uses **only** the content of a Student Handbook PDF and shows the source page for every answer.

📓 Notebook: [`rag_student_handbook_chatbot.ipynb`](rag_student_handbook_chatbot.ipynb)

## Architecture

```
            ┌──────────── Indexing (once) ────────────┐
 PDF ──► Extract text per page ──► Chunk ──► Embed ──► ChromaDB
            └─────────────────────────────────────────┘
            ┌──────────── Query (per question) ───────┐
 Question ──► Embed ──► Top-4 similar chunks ──► Prompt (context + question)
                                                    │
                                          Llama-3.3-70B (Groq)
                                                    │
                                     Answer + source page numbers
            └─────────────────────────────────────────┘
```

## Components

| Step | Tool | Details |
|---|---|---|
| Load PDF | `pypdf` | 69 pages extracted, whitespace cleaned, page numbers kept |
| Chunking | Custom function | 1000-character chunks, 200-character overlap, sentence-aware, giving 190 chunks |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, 384-dimensional, normalised |
| Vector store | `chromadb` | Collection `student_handbook` with page-number metadata |
| Retrieval | ChromaDB query | Top-k = 4 most similar chunks |
| Generation | `groq` | `llama-3.3-70b-versatile`, temperature 0.1, max 500 tokens |

The prompt tells the model to answer **only** from the retrieved context, to cite page numbers, and to reply *"The information is not available in the retrieved sections"* instead of guessing.

## Output Format

```
STUDENT QUESTION  ↓  RETRIEVED CONTEXT  ↓  GENERATED ANSWER  ↓  SOURCE / PAGE NUMBER
```

## Sample Results

| Question | Outcome | Source pages |
|---|---|---|
| Minimum attendance for the end-semester exam? | 75% (higher limits for some regulatory bodies) | 19, 20, 40 |
| How many days of academic leave? | Not available in the retrieved context (no hallucination) | 19, 37 |
| Rules for examination misconduct? | List of unfair-means acts and committee penalties | 23, 24, 25 |
| Library timings? | Not available in the retrieved context (no hallucination) | 8, 15, 23 |
| Disciplinary action for violating rules? | Partial answer grounded in context | 25, 26, 56 |

Every generated answer was checked against the retrieved pages. When the right passage was not retrieved, the model correctly refused to answer.

## Possible Improvements

- Hybrid search (BM25 + vectors) and cross-encoder re-ranking
- Section-aware or smaller chunks
- Persistent ChromaDB storage

## Requirements

```bash
pip install pypdf sentence-transformers chromadb groq
```
You need a free **Groq API key** from [console.groq.com](https://console.groq.com). It is entered securely at runtime with `getpass`.

## How to Run

1. Open the notebook in Google Colab.
2. Upload the Student Handbook PDF when prompted.
3. Enter your Groq API key when prompted.
4. Run the cells, then ask your own questions in the interactive loop at the end.
