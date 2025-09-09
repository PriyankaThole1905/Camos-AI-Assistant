# src/rag_pipeline.py
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.utils import load_prompt_templates

class CamosRAGPipeline:
    """
    Manages the LangChain RAG pipeline for Camos queries.
    """
    def __init__(self, config):
        self.config = config
        self.prompt_templates = load_prompt_templates("config/prompt_templates.yaml")

        # Initialize Embedding Model
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=self.config['embedding_model_name'],
            model_kwargs=self.config['embedding_model_kwargs']
        )
        print(f"Embeddings initialized with model: {self.config['embedding_model_name']}")

        # Initialize Ollama LLM
        self.llm = Ollama(
            base_url=self.config['ollama_base_url'],
            model=self.config['ollama_model_name'],
            temperature=self.config['ollama_temperature']
        )
        print(f"Ollama LLM initialized with model: {self.config['ollama_model_name']}")

        # Load/Create Vector Store (Done in data_ingestor, but needed here for retriever)
        self.vectorstore = self._load_or_recreate_vector_store()
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 documents
            print("Retriever initialized.")
        else:
            self.retriever = None
            print("WARNING: Vector store not loaded, RAG chain will not be fully functional.")

        # Initialize RAG Chain
        self.qa_chain = self._initialize_qa_chain()
        print("RAG Pipeline initialized.")

    def _load_or_recreate_vector_store(self):
        """Loads vector store or prompts user to ingest data if not found."""
        vector_store_path = self.config['vector_store_path']
        if os.path.exists(vector_store_path) and len(os.listdir(vector_store_path)) > 0:
            from src.data_ingestor import load_vector_store # Import locally to avoid circular dep
            return load_vector_store(
                self.config['embedding_model_name'],
                self.config['embedding_model_kwargs'],
                vector_store_path
            )
        else:
            print(f"FAISS index not found at {vector_store_path}. Please ingest data first.")
            return None

    def _initialize_qa_chain(self):
        """Initializes the RetrievalQA chain."""
        if not self.retriever:
            return None
        
        # Create prompt template for RAG
        rag_prompt = PromptTemplate.from_template(self.prompt_templates['rag_template'])

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # 'stuff' combines all docs into one prompt
            retriever=self.retriever,
            return_source_documents=True, # To show which documents were used
            chain_type_kwargs={"prompt": rag_prompt}
        )
        return qa_chain

    def query_rag(self, question):
        """Queries the RAG pipeline."""
        if not self.qa_chain:
            return "RAG system not ready. Please ensure data has been ingested."
        
        try:
            result = self.qa_chain({"query": question})
            return result['result']
        except Exception as e:
            print(f"Error during RAG query: {e}")
            return f"An error occurred during AI processing: {e}. Please ensure Ollama server is running and the model '{self.config['ollama_model_name']}' is pulled."

    def debug_code(self, code_snippet, error_message):
        """Uses Ollama directly to debug a code snippet."""
        # This bypasses RAG for a direct LLM call focused on debugging
        debug_prompt = self.prompt_templates['debug_template'].format(
            code_snippet=code_snippet,
            error_message=error_message
        )
        try:
            return self.llm.invoke(debug_prompt)
        except Exception as e:
            print(f"Error during code debugging: {e}")
            return f"An error occurred during code debugging: {e}. Please ensure Ollama server is running and the model '{self.config['ollama_model_name']}' is pulled."