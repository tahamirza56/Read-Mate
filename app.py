"""
PDF Q&A Application using RAG (Retrieval-Augmented Generation)
==============================================================
A clean, portfolio-ready application that allows users to upload PDFs
and ask questions about their content using Google Gemini AI.

Architecture:
- Streamlit for UI
- LangChain for orchestration
- PyPDF2 for PDF processing
- FAISS for vector storage
- Google Gemini for embeddings and LLM
"""

import os
import tempfile
import time
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Optional

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_css():
    """Inject custom CSS for a polished, modern UI."""
    st.markdown("""
    <style>
    /* ===== Global Layout ===== */
    .main > div {
        padding-top: 1.5rem;
        background-color: #2b1515 !important;
    }
    .stApp {
        background-color: #2b1515 !important;
    }

    header[data-testid="stHeader"] {
        background-color: #2b1515 !important;
    }


    /* ===== Header ===== */
    .app-header {
        background: #2b1515 !important;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: white !important;
    }
    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
        color: white !important;
    }

    /* ===== Sidebar ===== */
    .css-1d391kg {
        background: #242121 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background: #242121 !important;
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #e0e0e0 !important;
    }
    /* Sidebar branding */
    section[data-testid="stSidebar"] h2 {
        color: #3095e3 !important;
    }
    section[data-testid="stSidebar"] p {
        color: #8a9ae8 !important;
    }

    /* ===== Section Headers in Sidebar ===== */
    .sidebar-section {
        background: rgba(48, 149, 227, 0.1) !important;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(48, 149, 227, 0.3);
    }
    .sidebar-section h3 {
        color: #3095e3 !important;
        margin: 0 0 0.75rem 0;
        font-size: 1.1rem;
    }

    /* ===== File Uploader ===== */
    .stFileUploader > div:first-child {
        border: 2px dashed #3095e3 !important;
        border-radius: 12px !important;
        background: rgba(48, 149, 227, 0.15) !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div:first-child:hover {
        border-color: #3095e3 !important;
        background: rgba(48, 149, 227, 0.25) !important;
    }
    /* Upload button (Browse files) */
    .stFileUploader button[kind="secondary"] {
        background: black !important;
        color: white !important;
        border: 1px solid #3095e3 !important;
    }
    .stFileUploader button[kind="secondary"]:hover {
        background: #1a1a1a !important;
        border-color: #4aa3e8 !important;
    }

    /* ===== Chat Input ===== */
    .stChatInput {
        border-radius: 12px !important;
        border: 2px solid #3095e3 !important;
        background: rgba(27, 21, 21, 0.8) !important;
    }
    .stChatInput:focus-within {
        border-color: #4aa3e8 !important;
    }
    .stChatInput input {
        color: #2e2424 !important;
        background: transparent !important;
    }
    .stChatInput textarea {
        color: #2e2424 !important;
        background: transparent !important;
    }
    /* Placeholder text color */
    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder {
        color: #8a9ae8 !important;
        opacity: 1 !important;
    }

    /* ===== Chat Messages ===== */
    .stChatMessage {
        background: rgba(48, 149, 227, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        color: white !important;
        border: 1px solid rgba(48, 149, 227, 0.2);
    }
    .stChatMessage p {
        color: white !important;
    }

    /* ===== User Message ===== */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: linear-gradient(135deg, #3095e3 0%, #1a0d0d 100%) !important;
        color: white !important;
        border: 1px solid rgba(48, 149, 227, 0.5);
    }
    .stChatMessage[data-testid="stChatMessageUser"] p {
        color: white !important;
    }

    /* ===== Assistant Message ===== */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: rgba(48, 149, 227, 0.15) !important;
        border-left: 4px solid #3095e3 !important;
        color: white !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] p {
        color: white !important;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background: #3095e3 !important;
        color: white !important;
        border: 1px solid #3095e3 !important;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(48, 149, 227, 0.5) !important;
        background: #4aa3e8 !important;
        border-color: #4aa3e8 !important;
    }

    /* ===== Expander (Sources) ===== */
    .streamlit-expanderHeader {
        background: rgba(48, 149, 227, 0.15) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #3095e3 !important;
        border: 1px solid rgba(48, 149, 227, 0.3);
    }
    .streamlit-expanderContent {
        background: rgba(48, 149, 227, 0.08) !important;
        border-radius: 0 0 8px 8px !important;
        color: white !important;
        border: 1px solid rgba(48, 149, 227, 0.2);
        border-top: none;
    }

    /* ===== Metrics Cards ===== */
    .metric-card {
        background: rgba(48, 149, 227, 0.1) !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        text-align: center;
        border-top: 3px solid #3095e3;
        color: white !important;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #3095e3;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #e0e0e0;
        margin-top: 0.25rem;
    }

    /* ===== Welcome Screen ===== */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background: #2b1515 !important;
        border-radius: 16px;
        border: 1px solid rgba(48, 149, 227, 0.3);
    }
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .welcome-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3095e3 !important;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        font-size: 1.05rem;
        color: #b0b8e8 !important;
        margin-bottom: 2rem;
    }

    /* ===== Source Citation Cards ===== */
    .source-card {
        background: rgba(48, 149, 227, 0.1) !important;
        border-left: 3px solid #3095e3;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: white !important;
    }
    .source-page {
        display: inline-block;
        background: #3095e3;
        color: white;
        padding: 0.1rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ===== Progress Bar ===== */
    .stProgress > div > div {
        background: linear-gradient(135deg, #3095e3 0%, #1a0d0d 100%) !important;
    }

    /* ===== Info/Success/Error Boxes ===== */
    .stAlert {
        border-radius: 10px !important;
        background: rgba(48, 149, 227, 0.15) !important;
        color: white !important;
        border: 1px solid rgba(48, 149, 227, 0.3);
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #8a9ae8;
        font-size: 0.85rem;
        border-top: 1px solid rgba(48, 149, 227, 0.2);
        margin-top: 2rem;
    }

    /* ===== Spinner ===== */
    .stSpinner {
        color: #3095e3 !important;
    }

    /* ===== Divider ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(48, 149, 227, 0.5), transparent);
        margin: 1.5rem 0;
    }

    /* ===== Text inputs ===== */
    .stTextInput input {
        background: rgba(48, 149, 227, 0.1) !important;
        color: #2e2c2c !important;
        border: 1px solid rgba(48, 149, 227, 0.5) !important;
    }
    .stTextInput label {
        color: #2e2c2c !important;
    }

    /* ADD THIS BLOCK to make the eye icon black */
    section[data-testid="stSidebar"] .stTextInput button,
    section[data-testid="stSidebar"] .stTextInput button svg {
        color: black !important;
        fill: black !important;

    /* ===== Select box ===== */
    .stSelectbox > div > div {
        background: rgba(48, 149, 227, 0.1) !important;
        color: white !important;
    }

    /* ===== General text ===== */
    .stMarkdown, .stText, p, span, div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_config() -> dict:
    """Load configuration from environment variables with defaults."""
    return {
        "model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "top_k": int(os.getenv("TOP_K", "4")),
    }


# ============================================================================
# PDF PROCESSING
# ============================================================================

def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List:
    """
    Load a PDF file and split it into chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of each chunk (characters)
        chunk_overlap: Overlap between chunks for context continuity

    Returns:
        List of Document objects containing the chunks
    """
    # Load PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Initialize text splitter with recursive character splitting
    # This splitter tries to split on paragraphs, sentences, then characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)

    return chunks


# ============================================================================
# VECTOR STORE
# ============================================================================

def create_vector_store(
    chunks: List,
    api_key: str,
    embedding_model: str = "models/gemini-embedding-001",
    max_retries: int = 5
) -> FAISS:
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks: List of Document objects to embed
        api_key: Google API key for Gemini embeddings
        embedding_model: Name of the Gemini embedding model
        max_retries: Number of retry attempts for transient API errors

    Returns:
        FAISS vector store containing the embedded chunks
    """
    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key,
    )

    # Retry loop to handle transient 500/429 errors from the embedding API
    # Google's free tier occasionally returns INTERNAL errors that succeed on retry
    last_error = None
    for attempt in range(max_retries):
        try:
            # Create FAISS vector store from chunks
            # This will embed each chunk and store it in the FAISS index
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            return vector_store

        except Exception as e:
            last_error = e
            error_str = str(e)

            # Only retry on transient server errors (500) or rate limits (429)
            if "500" in error_str or "INTERNAL" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s, 16s
                    wait_time = 2 ** (attempt + 1)
                    st.warning(
                        f"⏳ Embedding API busy (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
            else:
                # Non-transient error (e.g. auth, bad model) — fail immediately
                raise

    # If we exhausted all retries, raise the last error
    raise last_error


# ============================================================================
# QA CHAIN
# ============================================================================

def create_qa_chain(
    vector_store: FAISS,
    api_key: str,
    model_name: str = "gemini-3.6-flash",
    top_k: int = 4
) -> ConversationalRetrievalChain:
    """
    Create a conversational retrieval chain for Q&A.

    Args:
        vector_store: FAISS vector store to retrieve from
        api_key: Google API key for Gemini LLM
        model_name: Name of the Gemini model to use
        top_k: Number of relevant chunks to retrieve

    Returns:
        ConversationalRetrievalChain for answering questions
    """
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,  # Lower temperature for more deterministic answers
        convert_system_message_to_human=True,
    )

    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    # Custom prompt template for strict context-based answering
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context.
    Always answer based on the context provided. If the answer is not in the context,
    say "I don't have enough information to answer this question based on the provided document."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Initialize conversation memory
    # This keeps track of the chat history for contextual follow-up questions
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    return chain


def invoke_with_retry(chain, inputs: dict, max_retries: int = 3) -> dict:
    """
    Invoke the QA chain with retry logic for transient API errors.

    Args:
        chain: The ConversationalRetrievalChain to invoke
        inputs: Dictionary with 'question' and 'chat_history'
        max_retries: Number of retry attempts for transient errors

    Returns:
        Result dictionary from the chain
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_error = e
            error_str = str(e)

            # Only retry on transient server errors (500) or rate limits (429)
            if "500" in error_str or "INTERNAL" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    st.warning(
                        f"⏳ AI service busy (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
            else:
                # Non-transient error — fail immediately
                raise

    # If we exhausted all retries, raise the last error
    raise last_error


# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None


def display_sidebar():
    """Display the sidebar with PDF upload and API key input."""
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem;">📄</div>
            <h2 style="color: #a0a8ff; margin: 0.5rem 0 0 0;">ReadMate</h2>
            <p style="color: #888; font-size: 0.85rem; margin: 0;">RAG System</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ===== API Key Section =====
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 🔑 API Configuration")
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Get yours at: https://makersuite.google.com/app/apikey",
                placeholder="AIza...",
                label_visibility="collapsed"
            )

            # Load from .env if available
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")

            if api_key:
                st.session_state.api_key = api_key
                st.success("✅ API Key loaded")
            else:
                st.warning("⚠️ Add your API key to start")
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== Upload Section =====
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 📤 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                help="Upload a PDF to ask questions about its content",
                label_visibility="collapsed"
            )

            if uploaded_file and api_key:
                process_pdf(uploaded_file, api_key)

            # Show document info if loaded
            if st.session_state.vector_store:
                st.divider()
                st.markdown("### 📊 Document Status")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Status", "✅ Ready")
                with col_b:
                    st.metric("Chunks", st.session_state.num_chunks)
                st.caption(f"📄 {st.session_state.doc_name}")
                st.caption("Ask questions in the chat →")
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== Tips Section =====
        if st.session_state.vector_store:
            with st.container():
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("### 💡 Tips")
                st.markdown("""
                - Ask specific questions
                - Reference page numbers
                - Try follow-up questions
                - Check sources for context
                """)
                st.markdown('</div>', unsafe_allow_html=True)


def process_pdf(uploaded_file, api_key: str):
    """Process the uploaded PDF file."""
    # Check if this is a new file or already processed
    if st.session_state.get("last_file") != uploaded_file.name:
        # Create progress container
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Save file
            status_text.text("📥 Saving uploaded file...")
            progress_bar.progress(10)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Step 2: Extract text
            status_text.text("📖 Extracting text from PDF...")
            progress_bar.progress(30)

            config = get_config()
            chunks = load_and_split_pdf(
                tmp_path,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )
            st.session_state.num_chunks = len(chunks)
            st.session_state.doc_name = uploaded_file.name

            # Step 3: Create embeddings
            status_text.text("🧠 Generating embeddings...")
            progress_bar.progress(55)

            vector_store = create_vector_store(
                chunks,
                api_key=api_key,
                embedding_model=config["embedding_model"]
            )

            # Step 4: Build QA chain
            status_text.text("🔗 Building QA chain...")
            progress_bar.progress(80)

            qa_chain = create_qa_chain(
                vector_store,
                api_key=api_key,
                model_name=config["model"],
                top_k=config["top_k"]
            )

            # Store in session state
            st.session_state.vector_store = vector_store
            st.session_state.qa_chain = qa_chain
            st.session_state.last_file = uploaded_file.name
            st.session_state.chat_history = []

            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Ready!")
            time.sleep(0.5)
            progress_container.empty()
            status_text.empty()

            st.success(f"✅ Processed {len(chunks)} chunks successfully!")

        except Exception as e:
            progress_container.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def display_welcome():
    """Display welcome screen when no document is loaded."""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">Welcome to ReadMate</div>
        <div class="welcome-subtitle">
            Upload a PDF document and ask questions about its content.
        </div>
        <br>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">📄</div>
            <div class="value">Upload</div>
            <div class="label">Drag & drop any PDF</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🔍</div>
            <div class="value">Search</div>
            <div class="label">AI-powered retrieval</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">💬</div>
            <div class="value">Ask</div>
            <div class="label">Natural language Q&A</div>
        </div>
        """, unsafe_allow_html=True)


def display_chat():
    """Display the chat interface and handle user interactions."""

    # If no document loaded, show welcome screen
    if not st.session_state.qa_chain:
        display_welcome()
        return

    # Chat header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">💬</span>
        <h3 style="margin: 0; color: #2a2a3e;">Chat with your document</h3>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 View {len(message['sources'])} sources"):
                    for j, doc in enumerate(message["sources"], 1):
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="source-page">Page {page}</span>
                            <p style="margin: 0.5rem 0 0 0;">{doc.page_content[:400]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if j < len(message["sources"]):
                            st.divider()

    # Example questions
    if not st.session_state.chat_history:
        st.markdown("### 💭 Try asking:")
        example_questions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the conclusions?",
        ]
        cols = st.columns(3)
        for idx, q in enumerate(example_questions):
            with cols[idx]:
                if st.button(q, key=f"example_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": q
                    })
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Convert chat history to format expected by LangChain
                    history = []
                    for msg in st.session_state.chat_history[:-1]:  # Exclude the current message
                        if msg["role"] == "user":
                            history.append((msg["content"], ""))
                        elif msg["role"] == "assistant":
                            if history:
                                history[-1] = (history[-1][0], msg["content"])

                    # Get answer from QA chain (with retry for transient errors)
                    result = invoke_with_retry(
                        st.session_state.qa_chain,
                        {"question": prompt, "chat_history": history}
                    )

                    answer = result["answer"]
                    source_documents = result.get("source_documents", [])

                    # Display answer
                    st.write(answer)

                    # Display sources
                    if source_documents:
                        with st.expander(f"📚 View {len(source_documents)} sources"):
                            for i, doc in enumerate(source_documents, 1):
                                page = doc.metadata.get('page', 'N/A')
                                st.markdown(f"""
                                <div class="source-card">
                                    <span class="source-page">Page {page}</span>
                                    <p style="margin: 0.5rem 0 0 0;">{doc.page_content[:400]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                                if i < len(source_documents):
                                    st.divider()

                    # Add assistant message to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_documents
                    })

                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_message
                    })


def main():
    """Main function to run the Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="PDF Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_css()

    # Initialize session state
    init_session_state()

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>PDF Assistant</h1>
        <p>Upload documents and chat with them using AI-powered Retrieval-Augmented Generation</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout: Sidebar + Main chat area
    col1, col2 = st.columns([1, 3])

    with col1:
        display_sidebar()

    with col2:
        display_chat()

    # Footer
    st.markdown("""
    <div class="footer">
    
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()