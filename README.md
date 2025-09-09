# 🤖 Camos AI Assistant

The **Camos AI Assistant** is an intelligent knowledge management and support system built for developers and engineers working with **Camos CPQ (Configure, Price, Quote)** software.  
It combines **document ingestion, OCR, semantic search, and RAG (Retrieval-Augmented Generation)** with a **Streamlit web interface** to provide instant answers, debugging help, and a community-driven FAQ hub.  

---

## ✨ Features
- 📄 **Automated PDF Ingestion**
  - Extracts raw text using **PyMuPDF**.
  - Performs OCR on images with **Tesseract**.
  - Extracts structured data from tables using **Camelot**.
  - Splits documents into chunks with **LangChain Text Splitter**.

- 🔍 **Knowledge Base & Vector Search**
  - Embeds documents with **SentenceTransformer** models.
  - Stores semantic vectors in a **FAISS database**.
  - Efficient retrieval of top-k relevant documents.

- 💡 **AI-Powered Assistance**
  - Uses **Ollama LLMs (e.g., Mistral, Llama2)** for offline inference.
  - Supports **contextual RAG-based Q&A** with source citation.
  - Provides **code debugging assistance** (error + code analysis).

- 👥 **Community Hub**
  - FAQ system with search, contribution, and browsing.
  - Pending question queue for junior engineers.
  - Excel-based persistence of questions and answers.

- 🌐 **Streamlit Web UI**
  - Secure login with user details and experience level.
  - Tabs: **Chatbot 💬 | FAQs ❓ | Ask & Answer ✍️**
  - Sidebar controls for **knowledge base ingestion**.
  - Real-time chat with memory support.

---

## 🛠️ Tech Stack
- **Python**  
- **LangChain** – RAG pipeline, document splitting  
- **FAISS** – Vector database  
- **SentenceTransformers** – Embedding model  
- **Ollama** – Local LLM hosting  
- **PyMuPDF, Tesseract OCR, Camelot** – Document parsing  
- **Streamlit** – Web interface  
- **Pandas, OpenPyXL** – Excel-based data persistence  
- **YAML** – Config & prompt templates  

---

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/camos-ai-assistant.git
cd camos-ai-assistant




