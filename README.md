# 📄 RAG Document Chatbot

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from any PDF document. It uses local embeddings for semantic search and a Hugging Face language model to generate answers – **completely offline after the first setup**.

---

## ✨ Features

- 📂 **Upload any PDF** – Ask questions about its content.
- 🔍 **Semantic Search** – Retrieves the most relevant document chunks using `all-MiniLM-L6-v2` embeddings.
- 🤖 **Local LLM** – Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (or any Hugging Face model) for answer generation – no internet required after the initial model download.
- ⚡ **Fast Retrieval** – ChromaDB vector store for efficient similarity search.
- 🖥️ **Simple CLI Interface** – Easy to use and extend.
- 🔒 **Privacy-First** – All processing happens on your local machine; your data never leaves your computer.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **LangChain** – For orchestration (document loading, splitting, chains)
- **ChromaDB** – Vector database
- **HuggingFace Transformers** – Embeddings and language model
- **PyPDF** – PDF parsing
- **python-dotenv** – Environment management

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/MISHU-KHONDOKER/rag-document-chatbot.git
   cd rag-document-chatbot

2. Create and activate a virtual environment (recommended)

python -m venv venv
# On Windows:
.\venv\Scripts\Activate
# On Mac/Linux:
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

🚀 Usage

Run the chatbot: python rag_chatbot.py

### Example Interaction
![Chatbot Screenshot](screenshot.png)

🤖 RAG Chatbot Started...

