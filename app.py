import streamlit as st
from PIL import Image
import os
import tempfile
from uuid import uuid4
import numpy as np
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, PyMuPDFLoader
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


MAX_FILE_SIZE_MB = 10
DATA_FOLDER_PATH = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_FOLDER_PATH, exist_ok=True)


client = chromadb.PersistentClient(path=DATA_FOLDER_PATH)


text_embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
image_embedding_function = OpenCLIPEmbeddingFunction()


os.environ["GROQ_API_KEY"] = "your groq api key"


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def process_documents(documents, collection_name, embedding_function):
    collection = client.get_or_create_collection(name=collection_name)
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    ids = [str(uuid4()) for _ in docs]
    contents = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    embeddings = embedding_function.embed_documents(contents)
    collection.add(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)


def main():
    st.set_page_config(page_title="RAG Multi-Modal Chat", layout="wide")
    st.title("🤖 AI-Powered RAG Chat Application")

    with st.sidebar:
        st.header("Upload Your Files")
        uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        uploaded_texts = st.file_uploader("Upload Text Files", type="txt", accept_multiple_files=True)
        uploaded_csvs = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)

        if st.button("Process Files"):
            with st.spinner("Processing files...🤔"):
                if uploaded_pdfs:
                    pdf_docs = []
                    for file in uploaded_pdfs:
                        file_path = save_uploaded_file(file)
                        loader = PyMuPDFLoader(file_path)
                        pdf_docs.extend(loader.load())
                    process_documents(pdf_docs, "pdf_collection", text_embedding_function)

                if uploaded_texts:
                    text_docs = []
                    for file in uploaded_texts:
                        file_path = save_uploaded_file(file)
                        loader = TextLoader(file_path, encoding="utf-8")
                        text_docs.extend(loader.load())
                    process_documents(text_docs, "text_collection", text_embedding_function)

                if uploaded_csvs:
                    csv_docs = []
                    for file in uploaded_csvs:
                        file_path = save_uploaded_file(file)
                        loader = CSVLoader(file_path, encoding="utf-8")
                        csv_docs.extend(loader.load())
                    process_documents(csv_docs, "csv_collection", text_embedding_function)


                st.success("Files processed successfully!")

    st.header("Query Section")
    query = st.text_input("💬 Enter Your Query:", placeholder="Ask me anything AI-related!")

    query_style = st.radio("Select Query Style", ["To the Point", "Detailed", "Bullet Points"])

    if query:
      st.spinner("Thinking...🤔")
      query_type = st.radio("Query Type", ["Text Document", "PDF Collection","CSV Collection"],
       horizontal=True)

      if query_type == "Text Document":
         vectordb = Chroma(collection_name='text_collection', client=client, 
         embedding_function=text_embedding_function)
         retriver = vectordb.as_retriever()
         llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

         qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", 
         retriever=retriver, return_source_documents=True,)

         if query_style == "To the Point":
             concise_query = f"Answer briefly: {query}"
         elif query_style == "Detailed":
             concise_query = f"Provide a detailed answer: {query}"
         elif query_style == "Bullet Points":
             concise_query = f"Answer in bullet points: {query}"
         else:
              concise_query = query  

         result = qa_chain.invoke({"query": concise_query})
         st.write("### 🤖Answer:")
         st.markdown(
                f"""
                <div style="background-color: black; padding: 15px; border-radius: 10px; 
                border: 0px solid #cce7ff;">
                <p>{result["result"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        #  st.write(result["result"])
        

         st.write("###  📚References:")
         for doc in result["source_documents"]:
            html_content = f"""
                <div style="background-color: black; padding: 10px; border-radius: 10px; color: white;">
                <b>Document Name:</b> {doc.metadata.get('source')}<br>
                <b>Document ID:</b> {doc.metadata.get('id')}<br>
                <b>Page:</b> {doc.metadata.get('page')}<br>
                <b>Snippet:</b> {doc.page_content[:100]}...
                </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)


     
      elif query_type == "PDF Collection":
         vectordb = Chroma(collection_name='pdf_collection', client=client, embedding_function=text_embedding_function)
         retriver = vectordb.as_retriever()
         llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

         qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=retriver, return_source_documents=True,)

         if query_style == "To the Point":
             concise_query = f"Answer briefly: {query}"
         elif query_style == "Detailed":
             concise_query = f"Provide a detailed answer: {query}"
         elif query_style == "Bullet Points":
             concise_query = f"Answer in bullet points: {query}"
         else:
              concise_query = query  

         result = qa_chain.invoke({"query": concise_query})
         st.write("### Answer:")
         st.markdown(
                f"""
                <div style="background-color: black; padding: 15px; border-radius: 10px; border: 0px solid #cce7ff;">
                <p>{result["result"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
         st.write("###  📚References:")
         for doc in result["source_documents"]:
            html_content = f"""
                <div style="background-color: black; padding: 10px; border-radius: 10px; color: white;">
                <b>Document Name:</b> {doc.metadata.get('source')}<br>
                <b>Document ID:</b> {doc.metadata.get('id')}<br>
                <b>Page:</b> {doc.metadata.get('page')}<br>
                <b>Snippet:</b> {doc.page_content[:100]}...
                </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
            
      elif query_type == "CSV Collection":
         vectordb = Chroma(collection_name='csv_collection', client=client, embedding_function=text_embedding_function)
         retriver = vectordb.as_retriever()
         llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

         qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=retriver, return_source_documents=True,)

         if query_style == "To the Point":
             concise_query = f"Answer briefly: {query}"
         elif query_style == "Detailed":
             concise_query = f"Provide a detailed answer: {query}"
         elif query_style == "Bullet Points":
             concise_query = f"Answer in bullet points: {query}"
         else:
              concise_query = query  

         result = qa_chain.invoke({"query": concise_query})
         st.write("### Answer:")
         st.markdown(
                f"""
                <div style="background-color: black; padding: 15px; border-radius: 10px; border: 0px solid #cce7ff;">
                <p>{result["result"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
         st.write("###  📚References:")
         for doc in result["source_documents"]:
            html_content = f"""
                <div style="background-color: black; padding: 10px; border-radius: 10px; color: white;">
                <b>Document Name:</b> {doc.metadata.get('source')}<br>
                <b>Document ID:</b> {doc.metadata.get('id')}<br>
                <b>Page:</b> {doc.metadata.get('page')}<br>
                <b>Snippet:</b> {doc.page_content[:100]}...
                </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
if __name__ == "__main__":
    main()



# import tempfile
# import streamlit as st
# from PIL import Image
# import os
# from uuid import uuid4
# from langchain.vectorstores import Chroma
# import pandas as pd
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
# from langchain.docstore.document import Document
# import torch
# import chromadb
# from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
# from chromadb.utils.data_loaders import ImageLoader
# import numpy as np
# import time
# from tqdm import tqdm
# import os
# from IPython.display import display
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from langchain_chroma import Chroma
# import chromadb.utils.embedding_functions as embedding_functions
# import uuid

# current_dir = os.getcwd()
# data_folder_path = os.path.join(current_dir, "data")

# client = chromadb.PersistentClient(path=data_folder_path)


# text_embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# image_embedding_function = OpenCLIPEmbeddingFunction()

# os.environ["GROQ_API_KEY"]="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"

# def save_uploaded_file(uploaded_file):
#     """Save the uploaded file to a temporary location."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         return tmp_file.name 
    
# class HuggingFaceEmbeddingFunction:
#     def __init__(self, embedding_function):
#         self.embedding_function = embedding_function

#     def __call__(self, input: str):
#         embeddings = self.embedding_function.embed_documents([input])
#         return np.array(embeddings[0])  


# hugging_embedding = HuggingFaceEmbeddingFunction(text_embedding_function)

# def process_documents(documents, collection_name, embedding_function):
#     collection = client.get_or_create_collection(name=collection_name, embedding_function=hugging_embedding)
#     text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
#     docs = text_splitter.split_documents(documents)

#     ids = [f"{uuid4()}" for _ in docs]
#     contents = [doc.page_content for doc in docs]
#     metadatas = [doc.metadata for doc in docs]

#     embeddings = embedding_function.embed_documents(contents)
#     collection.add(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)
#     return collection


# def main():
#     st.set_page_config(page_title="RAG Multi-Modal Chat", layout="wide")

#     st.title("🤖 AI-Powered RAG Chat Application")


#     with st.sidebar:
#         st.header("Upload Your Files")
#         uploaded_pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
#         uploaded_texts = st.file_uploader("Upload Text Files", type="txt", accept_multiple_files=True)
#         uploaded_csvs = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True)
#         uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
#         if st.button("Process"):
#             with st.spinner("Processing files...🤔"):
#                 if uploaded_pdfs:
#                     pdf_docs = []
#                     for file in uploaded_pdfs:
#                         file_path = save_uploaded_file(file) 
#                         loader = PyMuPDFLoader(file_path)
#                         pdf_docs.extend(loader.load())  
#                     process_documents(pdf_docs, "pdf_collection", text_embedding_function)


#                 if uploaded_texts:
#                     text_docs = []
#                     for file in uploaded_texts:
#                         file_path = save_uploaded_file(file)  
#                         loader = TextLoader(file_path, encoding="utf-8")
#                         text_docs.extend(loader.load()) 
#                     process_documents(text_docs, "text_collection", text_embedding_function)

#                 if uploaded_csvs:
#                     csv_docs = []
#                     for file in uploaded_csvs:
#                         file_path = save_uploaded_file(file) 
#                         loader = CSVLoader(file_path, encoding="utf-8")
#                         csv_docs.extend(loader.load())  
#                     process_documents(csv_docs, "csv_collection", text_embedding_function)


#                 if uploaded_images:
#                     image_folder = os.path.join(current_dir, "images")
#                     os.makedirs(image_folder, exist_ok=True)


#                     for image_file in uploaded_images:
#                         image_path = os.path.join(image_folder, image_file.name)
#                         with open(image_path, "wb") as f:
#                             f.write(image_file.read())


#                     image_collection = client.get_or_create_collection(name="image_collection", 
#                     embedding_function=image_embedding_function)


#                     for image_file in os.listdir(image_folder):
#                         image_path = os.path.join(image_folder, image_file)
#                         pil_image = Image.open(image_path)
#                         image_array = np.array(pil_image)
#                         image_collection.add(ids=[image_file], images=[image_array])


#                 st.success("Files processed successfully!")

#     st.header("Query Section")
#     query = st.text_input("💬 Enter Your Query:", placeholder="Ask me anything AI-related!")

#     query_style = st.radio("Select Query Style", ["To the Point", "Detailed", "Bullet Points"])

#     if query:
#       st.spinner("Thinking...🤔")
#       query_type = st.radio("Query Type", ["Text Document", "PDF Collection","CSV Collection", "Images"],
#        horizontal=True)

#       if query_type == "Text Document":
#          vectordb = Chroma(collection_name='text_collection', client=client, 
#          embedding_function=text_embedding_function)
#          retriver = vectordb.as_retriever()
#          llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

#          qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", 
#          retriever=retriver, return_source_documents=True,)

#          if query_style == "To the Point":
#              concise_query = f"Answer briefly: {query}"
#          elif query_style == "Detailed":
#              concise_query = f"Provide a detailed answer: {query}"
#          elif query_style == "Bullet Points":
#              concise_query = f"Answer in bullet points: {query}"
#          else:
#               concise_query = query  

#          result = qa_chain.invoke({"query": concise_query})
#          st.write("### 🤖Answer:")
#          st.markdown(
#                 f"""
#                 <div style="background-color: black; padding: 15px; border-radius: 10px; 
#                 border: 0px solid #cce7ff;">
#                 <p>{result["result"]}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         #  st.write(result["result"])
        

#          st.write("###  📚References:")
#          for doc in result["source_documents"]:
#             st.markdown(
#                     f"""
                    
#                     <div style="background-color: black; padding: 10px; border-radius: 10px; border: 0px solid #b3d9b3;">
#                     <b>Document Name:</b> {doc.metadata.get('source')}<br>
#                     <b>Document ID:</b> {doc.metadata.get('id')}<br>
#                     <b>Page:</b> {doc.metadata.get('page')}<br>
#                     <b>Snippet:</b> {doc.page_content[:100]}...
            
#                     </div>
#                     <br>
#                     """,
#                     unsafe_allow_html=True,
#                 )
#             # st.write(f"**Document Name**: {doc.metadata.get('source')}")
#             # st.write(f"**Document ID**: {doc.metadata.get('id')}")
#             # st.write(f"**Page**: {doc.metadata.get('page')}")
#             # st.write(f"**Text Snippet**: {doc.page_content[:300]}...")  # Show the first 300 characters of the text snippet
#             # st.write("")

#       elif query_type == "Images":
#         collection = client.get_collection("image_collection")
#         results = collection.query(query_texts=[query], n_results=3)
#         image_folder = "images"
#         st.write("### Related Images:")
#         for image_id, distance in zip(results['ids'][0], results['distances'][0]):
#             image_path = os.path.join(image_folder, image_id)
#             if os.path.exists(image_path):
#                 image = Image.open(image_path)
#                 st.image(image, caption=f"Image ID: {image_id} - Distance: {distance:.4f}")
#             else:
#                 st.write(f"Image not found: {image_path}")

#       elif query_type == "PDF Collection":
#          vectordb = Chroma(collection_name='pdf_collection', client=client, embedding_function=text_embedding_function)
#          retriver = vectordb.as_retriever()
#          llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

#          qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=retriver, return_source_documents=True,)

#          if query_style == "To the Point":
#              concise_query = f"Answer briefly: {query}"
#          elif query_style == "Detailed":
#              concise_query = f"Provide a detailed answer: {query}"
#          elif query_style == "Bullet Points":
#              concise_query = f"Answer in bullet points: {query}"
#          else:
#               concise_query = query  

#          result = qa_chain.invoke({"query": concise_query})
#          st.write("### Answer:")
#          st.markdown(
#                 f"""
#                 <div style="background-color: black; padding: 15px; border-radius: 10px; border: 0px solid #cce7ff;">
#                 <p>{result["result"]}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
        
#          st.write("### References:")
#          for doc in result["source_documents"]:
#             st.markdown(
#                     f"""
                    
#                     <div style="background-color: black; padding: 10px; border-radius: 10px; border: 0px solid #b3d9b3;">
#                     <b>Document Name:</b> {doc.metadata.get('source')}<br>
#                     <b>Document ID:</b> {doc.metadata.get('id')}<br>
#                     <b>Page:</b> {doc.metadata.get('page')}<br>
#                     <b>Snippet:</b> {doc.page_content[:100]}...
#                     </div>
                    
#                     """,
#                     unsafe_allow_html=True,
#                 )
            
#       elif query_type == "CSV Collection":
#          vectordb = Chroma(collection_name='csv_collection', client=client, embedding_function=text_embedding_function)
#          retriver = vectordb.as_retriever()
#          llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

#          qa_chain = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=retriver, return_source_documents=True,)

#          if query_style == "To the Point":
#              concise_query = f"Answer briefly: {query}"
#          elif query_style == "Detailed":
#              concise_query = f"Provide a detailed answer: {query}"
#          elif query_style == "Bullet Points":
#              concise_query = f"Answer in bullet points: {query}"
#          else:
#               concise_query = query  

#          result = qa_chain.invoke({"query": concise_query})
#          st.write("### Answer:")
#          st.markdown(
#                 f"""
#                 <div style="background-color: black; padding: 15px; border-radius: 10px; border: 0px solid #cce7ff;">
#                 <p>{result["result"]}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
        
#          st.write("### References:")
#          for doc in result["source_documents"]:
#             st.markdown(
#                     f"""
                    
#                     <div style="background-color: black; padding: 10px; border-radius: 10px; border: 0px solid #b3d9b3;">
#                     <b>Document Name:</b> {doc.metadata.get('source')}<br>
#                     <b>Document ID:</b> {doc.metadata.get('id')}<br>
#                     <b>Page:</b> {doc.metadata.get('page')}<br>
#                     <b>Snippet:</b> {doc.page_content[:50]}...
#                     </div>
                    
#                     """,
#                     unsafe_allow_html=True,
#                 )


# if __name__ == "__main__":
#     main()
