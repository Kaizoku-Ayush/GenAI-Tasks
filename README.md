# Generative AI & Deep Learning Projects

Three hands-on projects covering computer vision, Retrieval-Augmented Generation and LLM fine-tuning. All of them build on pre-trained models instead of training from scratch.

| # | Project | What it does | Key techniques | Result |
|---|---|---|---|---|
| 01 | [Smart Waste Classification](01-smart-waste-classification/) | Classifies photos of waste into **Glass, Paper, Plastic, Metal** | Transfer learning (MobileNetV2), data augmentation, early stopping | 100% test accuracy |
| 02 | [RAG Student Handbook Chatbot](02-rag-student-handbook-chatbot/) | Answers questions from a PDF handbook with source page numbers | Chunking, embeddings, ChromaDB, Llama-3.3-70B via Groq | Grounded answers, no hallucination |
| 03 | [LLM Fine-Tuning with LoRA](03-llm-finetuning-lora/) | Fine-tunes TinyLlama-1.1B on a custom Q&A dataset | LoRA / PEFT, tokenization, Hugging Face Trainer | Only 0.10% of parameters trained |

## Tech Stack

- **Languages & notebooks:** Python, Jupyter / Google Colab
- **Deep learning:** TensorFlow / Keras, PyTorch
- **LLM tooling:** Hugging Face Transformers, Datasets, PEFT, Sentence-Transformers
- **Retrieval:** ChromaDB, PyPDF, Groq API
- **Data & visualisation:** NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Pillow

## Repository Structure

```
.
├── 01-smart-waste-classification/
│   ├── smart_waste_classification.ipynb
│   └── README.md
├── 02-rag-student-handbook-chatbot/
│   ├── rag_student_handbook_chatbot.ipynb
│   └── README.md
└── 03-llm-finetuning-lora/
    ├── llm_finetuning_lora.ipynb
    └── README.md
```

## How to Run

Each notebook is self-contained and was built for **Google Colab** (a GPU runtime is recommended for projects 01 and 03). Open the notebook, run the cells from top to bottom, and upload the input file when prompted. Each project's README lists its requirements.

---

**Made by Ayush Kumar**
