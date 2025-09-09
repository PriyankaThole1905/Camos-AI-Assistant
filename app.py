import streamlit as st
import os
import json
import time
import pandas as pd
import yaml
import base64
import time

# Import modules from src
from src.utils import load_config
from src.data_ingestor import load_and_process_camos_docs, create_and_save_vector_store, load_vector_store
from src.rag_pipeline import CamosRAGPipeline

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Camos AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration Loading and Validation ---
CONFIG_FILE = "config/model_config.yaml"
try:
    config_data = load_config(CONFIG_FILE)
    if config_data is None:
        st.error(f"Error: Configuration file '{CONFIG_FILE}' is empty or could not be parsed as YAML.")
        st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading configuration from '{CONFIG_FILE}': {e}")
    st.stop()

# --- FAQ & Pending Questions File Configuration ---
EXCEL_FAQ_FILE = config_data.get('excel_faq_file', "data/faqs.xlsx")
PENDING_QUESTIONS_FILE = "data/pending_questions.xlsx"
os.makedirs(os.path.dirname(EXCEL_FAQ_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PENDING_QUESTIONS_FILE), exist_ok=True)

# --- Excel Data Management Functions ---
def load_data_from_excel(filepath, columns):
    """Loads data from an Excel file, creating a new one if it doesn't exist."""
    if os.path.exists(filepath):
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.warning(f"Error loading {filepath}: {e}. Creating a new empty DataFrame.")
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_data_to_excel(df, filepath):
    """Saves a Pandas DataFrame to an Excel file."""
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
        return True
    except Exception as e:
        st.error(f"Error saving to {filepath}: {e}")
        return False

# --- Initialize RAG Pipeline (once) ---
@st.cache_resource
def initialize_rag_pipeline_cached(config_data_for_cache):
    try:
        pipeline = CamosRAGPipeline(config_data_for_cache)
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize RAG Pipeline: {e}. Please check Ollama server, model, and config.")
        return None

rag_pipeline = initialize_rag_pipeline_cached(config_data)

# --- Data Ingestion Function (called by sidebar button) ---
def ingest_camos_data_to_faiss():
    if rag_pipeline is None:
        st.error("RAG Pipeline not initialized. Cannot ingest data.")
        return

    st.info("Starting Camos data ingestion into FAISS...")
    os.makedirs(config_data.get('vector_store_path', "data/faiss_index"), exist_ok=True)
    all_documents = load_and_process_camos_docs(
        config_data.get('pdf_data_dir', "data/raw_pdfs"),
        config_data.get('chunk_size', 500),
        config_data.get('chunk_overlap', 50)
    )

    if not all_documents:
        st.warning("No documents were loaded or processed for ingestion.")
        return

    vectorstore = create_and_save_vector_store(
        all_documents,
        config_data.get('embedding_model_name', "sentence-transformers/all-MiniLM-L6-v2"),
        config_data.get('embedding_model_kwargs', {'device': 'cpu'}),
        config_data.get('vector_store_path', "data/faiss_index")
    )

    if vectorstore:
        st.success("Camos documentation successfully ingested into the FAISS knowledge base!")
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error("Failed to create/save vector store.")

# --- Login & Access Control Logic ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Experience level mapping for access control
EXPERIENCE_LEVELS = ["0-2yr", "3-5yr", "6yr and above"]
CAN_ANSWER_QUESTION = st.session_state.user_data.get('experience_level', '') in ["3-5yr", "6yr and above"]

def show_login_page():
    st.title("Welcome to Camos AI Assistant")
    st.markdown("Please log in to access the assistant.")
    with st.form("login_form"):
        name = st.text_input("Name")
        email = st.text_input("Email ID")
        experience_level = st.selectbox("Experience Level", EXPERIENCE_LEVELS)
        submitted = st.form_submit_button("Log In")
        if submitted:
            if name and email:
                st.session_state.logged_in = True
                st.session_state.user_data = {
                    "name": name,
                    "email": email,
                    "experience_level": experience_level,
                    "user_id": f"{email}_{time.time_ns()}"
                }
                st.success(f"Welcome, {name}! You are now logged in.")
                time.sleep(1) # Give a moment for the user to see the message
                st.rerun()
            else:
                st.error("Name and Email ID are required.")

def show_main_app():
    st.sidebar.title("Camos AI Assistant")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.user_data['name']}")
    st.sidebar.markdown(f"**Experience:** {st.session_state.user_data['experience_level']}")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())
    st.sidebar.markdown("---")
    
    st.sidebar.header("Knowledge Base Management")
    if st.sidebar.button("Ingest/Rebuild FAISS Index (Advanced)"):
        ingest_camos_data_to_faiss()

    st.sidebar.markdown("---")
    if rag_pipeline:
        st.sidebar.success("AI is ready! Ollama connected.")
    else:
         st.sidebar.error(
            "AI not ready. Check console for errors. Ensure Ollama server is running and the model "
            f"('{config_data.get('ollama_model_name', 'mistral')}') is pulled. "
            "Also, ensure FAISS index is built by clicking 'Ingest/Rebuild FAISS Index'."
        )
    st.sidebar.markdown(f"Current FAISS Index Path: `{config_data.get('vector_store_path', 'data/faiss_index')}`")

    st.title("🤖 Camos AI Assistant")
    st.markdown("""
    Welcome to your AI assistant for Camos programming!
    """)

    # --- Tabbed Interface ---
    tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "❓ FAQs", "✍️ Ask/Answer"])

    with tab1:
        st.header("Chat with the Camos Expert")
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you with Camos today?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a Camos question, provide code, or an error message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("AI is thinking..."):
                if rag_pipeline:
                    try:
                        if "error" in prompt.lower() and "code" in prompt.lower():
                            response = rag_pipeline.debug_code(
                                code_snippet="Please provide the code snippet you're working with.",
                                error_message="Please provide the exact error message."
                            )
                        else:
                            response = rag_pipeline.query_rag(prompt)

                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    except Exception as e:
                        error_msg = f"An error occurred during AI processing: {e}. " \
                                    "Please check the console for more details and ensure " \
                                    f"Ollama server is running with the model '{config_data.get('ollama_model_name', 'mistral')}'. " \
                                    "Also, ensure the FAISS index is built."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": "I apologize, an error occurred while processing your request. Please try again or check the server status."})
                else:
                    st.error("RAG Pipeline is not initialized. Please check the sidebar for instructions to set up the AI.")
                    st.session_state.messages.append({"role": "assistant", "content": "The AI is not ready yet. Please try again after the system is fully initialized."})

    with tab2:
        st.header("Community FAQs")
        faqs_df = load_data_from_excel(EXCEL_FAQ_FILE, ['id', 'question', 'answer', 'timestamp', 'created_by'])

        # FAQ Search functionality
        search_query = st.text_input("Search FAQs:", placeholder="e.g., 'case statement' or 'file handling'")
        
        if search_query:
            filtered_faqs_df = faqs_df[
                faqs_df['question'].str.contains(search_query, case=False, na=False) |
                faqs_df['answer'].str.contains(search_query, case=False, na=False)
            ]
        else:
            filtered_faqs_df = faqs_df
        
        if not filtered_faqs_df.empty:
            for index, faq in filtered_faqs_df.iterrows():
                with st.expander(f"❓ {faq['question']}"):
                    st.markdown(f"**Answer:** {faq['answer']}")
                    st.markdown(f"<small>Created by: {faq.get('created_by', 'Anonymous')} on {faq['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>", unsafe_allow_html=True)
        else:
            st.info("No FAQs match your search or have been contributed yet.")

    with tab3:
        st.header("Ask & Answer Questions")
        st.info("Ask a question for a community member to answer, or answer a pending question below.")
        
        # --- Submit a New Question ---
        st.subheader("Submit a New Question")
        with st.form("ask_question_form"):
            new_question = st.text_area("Your Question about Camos:", height=100)
            submitted_q = st.form_submit_button("Submit Question")
            if submitted_q and new_question:
                pending_df = load_data_from_excel(PENDING_QUESTIONS_FILE, ['id', 'question', 'timestamp', 'asked_by'])
                new_row = pd.DataFrame([{
                    'id': str(time.time_ns()),
                    'question': new_question.strip(),
                    'timestamp': pd.to_datetime('now'),
                    'asked_by': st.session_state.user_data['name']
                }])
                pending_df = pd.concat([new_row, pending_df], ignore_index=True)
                if save_data_to_excel(pending_df, PENDING_QUESTIONS_FILE):
                    st.success("Your question has been submitted for review and will appear below.")
                    st.rerun()
            elif submitted_q:
                st.error("Please enter a question.")
        
        # --- Answer Pending Questions ---
        st.subheader("Pending Questions")
        pending_df = load_data_from_excel(PENDING_QUESTIONS_FILE, ['id', 'question', 'timestamp', 'asked_by'])
        
        if not pending_df.empty:
            for index, pending_q in pending_df.iterrows():
                with st.expander(f"❓ {pending_q['question']}"):
                    st.markdown(f"<small>Asked by: {pending_q.get('asked_by', 'Anonymous')} on {pending_q['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>", unsafe_allow_html=True)
                    
                    if CAN_ANSWER_QUESTION:
                        with st.form(f"answer_form_{pending_q['id']}"):
                            answer = st.text_area("Your Answer:", height=150)
                            st.markdown("""
                            <small>For images/videos, please paste a public URL. The app does not support direct file uploads.</small>
                            """, unsafe_allow_html=True)
                            submitted_a = st.form_submit_button("Submit Answer")
                            
                            if submitted_a and answer:
                                # Add the new Q&A to the main FAQ list
                                faqs_df = load_data_from_excel(EXCEL_FAQ_FILE, ['id', 'question', 'answer', 'timestamp', 'created_by'])
                                new_faq_row = pd.DataFrame([{
                                    'id': pending_q['id'],
                                    'question': pending_q['question'],
                                    'answer': answer.strip(),
                                    'timestamp': pd.to_datetime('now'),
                                    'created_by': st.session_state.user_data['name']
                                }])
                                faqs_df = pd.concat([new_faq_row, faqs_df], ignore_index=True)
                                save_data_to_excel(faqs_df, EXCEL_FAQ_FILE)
                                
                                # Remove the answered question from the pending list
                                pending_df = pending_df.drop(index)
                                save_data_to_excel(pending_df, PENDING_QUESTIONS_FILE)
                                
                                st.success("Answer submitted and added to FAQs!")
                                st.rerun()
                            elif submitted_a:
                                st.error("Please provide an answer.")
                    else:
                        st.info("You must have 3+ years of experience to answer questions.")
        else:
            st.info("No pending questions at this time.")

# --- Main app flow: show login or main app ---
if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page()
