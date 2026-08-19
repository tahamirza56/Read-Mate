# ReadMate - RAG Application

A clean, portfolio-ready PDF Question & Answer application built with Retrieval-Augmented Generation (RAG) architecture. Upload any PDF and ask questions about its content using Google Gemini AI.

## Features

- **PDF Upload & Processing**: Extract text from PDF documents using PyPDF2
- **Intelligent Chunking**: Split documents into optimal chunks for retrieval using LangChain's RecursiveCharacterTextSplitter
- **Vector Embeddings**: Generate embeddings using Google Gemini's embedding-001 model
- **Fast Retrieval**: Store and search embeddings locally using FAISS
- **RAG-powered Q&A**: Answer questions using retrieved context with Gemini LLM
- **Chat History**: Maintain conversation context for follow-up questions
- **Source Attribution**: Show which parts of the document were used to answer each question
- **Clean UI**: Modern Streamlit interface with sidebar configuration

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                                  │
│  ┌──────────────┐  ┌─────────────────────────────────────────────┐  │
│  │   Sidebar    │  │              Main Chat Area                  │  │
│  │ - API Key    │  │  - User questions                           │  │
│  │ - PDF Upload │  │  - AI responses                             │  │
│  │ - Doc Info   │  │  - Source citations                         │  │
│  └──────────────┘  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangChain Orchestration                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐│
│  │  PDF Loader    │  │ Text Splitter  │  │  Conversational Chain   ││
│  │  (PyPDF2)      │──│ (Recursive)    │──│  - Retriever            ││
│  └────────────────┘  └────────────────┘  │  - LLM (Gemini)        ││
│                                          │  - Memory               ││
│                                          └────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Google Gemini API                                 │
│  ┌────────────────────────┐  ┌────────────────────────────────┐    │
│  │  Embedding Model       │  │  Language Model                 │    │
│  │  (embedding-001)       │  │  (gemini-1.5-flash/pro)        │    │
│  └────────────────────────┘  └────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.9+
- Google Gemini API key (free tier available)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pdf-qa-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Configuration

The application can be configured via environment variables in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `GEMINI_MODEL` | LLM model to use | `gemini-1.5-flash` |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | `models/embedding-001` |
| `CHUNK_SIZE` | Maximum chunk size (chars) | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K` | Number of chunks to retrieve | `4` |

## Getting a Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key and add it to your `.env` file

## How It Works

1. **PDF Upload**: User uploads a PDF through the Streamlit interface
2. **Text Extraction**: PyPDFLoader extracts text from all pages
3. **Chunking**: RecursiveCharacterTextSplitter splits text into optimal chunks
4. **Embedding**: Gemini embedding-001 model creates vector embeddings
5. **Indexing**: FAISS stores embeddings for fast similarity search
6. **Query Processing**: User question is embedded and searched against index
7. **Retrieval**: Most relevant chunks are retrieved (top_k)
8. **Generation**: Gemini LLM generates answer based on retrieved context
9. **Response**: Answer and source citations are displayed to user

## Project Structure

```
pdf-qa-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── README.md             # Project documentation
└── .gitignore           # Git ignore file (recommended to add)
```

## Usage Tips

- **Large PDFs**: For very large documents (100+ pages), processing may take a few minutes
- **Chunk Size**: Larger chunks provide more context but may reduce precision
- **Model Selection**: Use `gemini-1.5-flash` for faster responses, `gemini-1.5-pro` for more complex questions
- **Question Quality**: Be specific in your questions for better results

## Common Issues

**API Key Error**
- Ensure your API key is correctly set in `.env` file
- Check that the API key has access to Gemini models

**PDF Processing Error**
- Ensure the PDF is not password-protected
- Try a smaller PDF file if memory issues occur

**Slow Processing**
- First query may be slow as embeddings are generated
- Subsequent queries are faster due to FAISS indexing

## License

MIT License - feel free to use this project for your portfolio!

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Gemini](https://ai.google.dev/)