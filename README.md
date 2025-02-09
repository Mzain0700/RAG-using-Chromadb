# AI-Powered RAG Chat Application
## 📌 Overview
This project is an AI-powered Retrieval-Augmented Generation (RAG) multi-modal chat application that allows users to upload various types of documents (PDFs, text files, CSVs) and retrieve relevant information using advanced AI models.
The project consists of two main files:
- ChromaDB,Langchain.ipynb → This file is responsible for processing and storing the document embeddings using ChromaDB and Langchain.
- app.py → This file runs the Streamlit-based GUI where users can upload files, enter queries, and receive AI-generated responses.

## 🚀 Features
- Multi-modal Document Processing: Supports PDFs, text files, and CSVs.
- AI-powered Search & Querying: Uses Langchain and ChatGroq for document retrieval.
- ChromaDB Integration: Efficient storage and retrieval of vectorized document embeddings.
- Customizable Query Style: Users can choose between To the Point, Detailed, or Bullet Points responses.
- Interactive Web Interface: Built using Streamlit.

## 📂 File Descriptions
### 1️⃣ ChromaDB,Langchain.ipynb (Backend Processing)
This Jupyter Notebook handles:
- Document Processing & Embedding
   - Loads documents (PDFs, text files, CSVs)
   - Splits text into manageable chunks
   - Generates vector embeddings using HuggingFaceEmbeddings
   - Stores embeddings in ChromaDB
- ChromaDB Vector Store
  - Stores processed documents as vectors for fast retrieval.
- Retrieval Pipeline
  - Uses Langchain retrievers to fetch relevant documents for queries.
#### 💡 Technologies Used:
- Langchain (for NLP & AI-powered search)
- ChromaDB (for document vector storage)
- HuggingFace Transformers (for text embeddings)

### 2️⃣ app.py (Frontend & GUI)
This is the Streamlit-based GUI that allows users to:
- Upload PDFs, text files, and CSVs.
- Process and store these files in ChromaDB.
- Ask queries and receive AI-generated responses based on document content.
- View references and sources from retrieved documents.

#### 🔹 How It Works?
- Users upload documents from the sidebar.
- Documents are processed, embedded, and stored in ChromaDB.
- Users enter a query in the text box.
- The system fetches relevant document snippets and generates a response using ChatGroq.
- The response is displayed in an interactive UI, along with source references.

#### 🛠 Key Libraries Used:
- Streamlit → For the user interface.
- Langchain → For document processing & querying.
- ChromaDB → For storing document vectors.
- ChatGroq → AI model for generating responses.
- Pillow, PyMuPDF, CSVLoader → For handling different document formats.

## 🔧 Setup & Installation
### 1️⃣ Install Required Libraries
``` pip install streamlit langchain chromadb huggingface_hub pymupdf pandas ```

### 2️⃣ Run the Application
``` streamlit run app.py ```

## 📝 Conclusion
This project is a powerful AI-based document search and retrieval system using ChromaDB, Langchain, and Streamlit. It allows users to upload files, query them using AI, and retrieve accurate responses with references.
####  💡 Feel free to contribute and improve this project! 🚀
