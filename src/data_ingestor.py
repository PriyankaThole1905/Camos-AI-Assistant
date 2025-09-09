# src/data_ingestor.py
import os
import fitz # PyMuPDF
from PIL import Image
import pytesseract
import io
import camelot
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Set the path to the Tesseract executable if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example for macOS
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

def extract_text_from_image_ocr(image_bytes):
    """Extracts text from an image using Tesseract OCR."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def extract_additional_content_from_pdf(filepath):
    """
    Extracts text from images and tables from a PDF using OCR and Camelot.
    Returns a list of Document objects with extracted content.
    """
    additional_docs = []
    try:
        document = fitz.open(filepath)
        filename_base = os.path.basename(filepath)

        # 1. OCR from Images
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                if image_ext in ["png", "jpeg", "jpg"]:
                    text_from_image = extract_text_from_image_ocr(image_bytes)
                    if text_from_image.strip():
                        additional_docs.append(Document(
                            page_content=f"Image content from page {page_num + 1}:\n{text_from_image}",
                            metadata={"source": filename_base, "page": page_num + 1, "type": "image_ocr", "image_index": img_index + 1}
                        ))
        
        # 2. Table Extraction with Camelot
        tables = camelot.read_pdf(filepath, pages='all', flavor='stream', line_scale=40)
        for i, table in enumerate(tables):
            df = table.df
            table_markdown = f"Table {i+1} from page {table.page}:\n\n"
            table_markdown += df.to_markdown(index=False)
            additional_docs.append(Document(
                page_content=table_markdown,
                metadata={"source": filename_base, "page": int(table.page), "type": "table_data", "table_index": i + 1}
            ))

        print(f"Successfully extracted additional content (OCR/Tables) from {filename_base}")
    except Exception as e:
        print(f"Error extracting additional content from {filename_base}: {e}")
    return additional_docs

def load_and_process_camos_docs(pdf_dir, chunk_size, chunk_overlap):
    """
    Loads PDFs, extracts content (text, OCR, tables), chunks them,
    and returns a list of LangChain Document objects.
    """
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file_path in pdf_files:
        try:
            # Load main text content using PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_file_path)
            docs = loader.load()
            
            # Extract additional content (OCR from images, tables)
            additional_docs = extract_additional_content_from_pdf(pdf_file_path)
            
            # Combine and chunk all documents
            combined_docs = docs + additional_docs
            chunked_docs = text_splitter.split_documents(combined_docs)
            all_documents.extend(chunked_docs)
            print(f"Processed and chunked {os.path.basename(pdf_file_path)} into {len(chunked_docs)} chunks.")
        except Exception as e:
            print(f"Error processing PDF {os.path.basename(pdf_file_path)}: {e}")
            
    return all_documents

def create_and_save_vector_store(documents, embedding_model_name, embedding_model_kwargs, vector_store_path):
    """
    Creates embeddings for documents and saves them to a FAISS vector store.
    """
    if not documents:
        print("No documents to add to vector store.")
        return None

    print(f"Creating embeddings with model: {embedding_model_name}")
    embeddings = SentenceTransformerEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs
    )
    
    print(f"Creating and saving FAISS vector store to {vector_store_path}")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(vector_store_path)
    print("Vector store created and saved successfully.")
    return vectorstore

def load_vector_store(embedding_model_name, embedding_model_kwargs, vector_store_path):
    """
    Loads an existing FAISS vector store.
    """
    if not os.path.exists(vector_store_path):
        print(f"FAISS index not found at {vector_store_path}.")
        return None

    print(f"Loading embeddings with model: {embedding_model_name}")
    embeddings = SentenceTransformerEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs
    )
    
    print(f"Loading FAISS vector store from {vector_store_path}")
    vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True) # allow_dangerous_deserialization is needed for newer FAISS versions
    print("Vector store loaded successfully.")
    return vectorstore