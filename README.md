# RAG Bot — Resume Intelligence

A powerful AI-driven web application that combines **Retrieval-Augmented Generation (RAG)** with resume optimization and company research capabilities. Built for AI/ML engineering students preparing for interviews at top tech companies.

## 🎯 Features

### 1. **Chat Interface**
- Ask questions about your resume content using natural language
- Retrieves relevant information from your PDF resume using vector embeddings
- Powered by Google's Gemini 2.5 Flash LLM with RAG capabilities

### 2. **Resume Modifier**
- Automatically tailors your resume to job descriptions
- Extracts keywords from job postings and incorporates them naturally
- Preserves original structure and content integrity
- Generates PDF output ready for submission

### 3. **Company Research**
- Fetches live information about company projects and tech stack
- Analyzes company methodologies and architecture patterns
- Identifies edge topics and niche technical areas
- Tailored interview preparation insights for specific companies

## 📋 Prerequisites

### Required API Keys
- **Google API Key** - For Gemini LLM and embeddings (set as environment variable: `GOOGLE_API_KEY`)
- **Tavily API Key** - For company research and web search (configure in code)

### Python Version
- Python 3.8 or higher

## 🚀 Installation

### 1. Clone or setup the project
```bash
cd "c:\Users\LENOVO\Downloads\RAG"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `langchain` - LLM orchestration framework
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - Hugging Face embeddings
- `langchain-community` - Document loaders and vector stores
- `faiss-cpu` - Vector similarity search
- `reportlab` - PDF generation
- `flask` - Web server
- `tavily-python` - Web search API client

### 3. Configure environment
```bash
# Set your Google API key
set GOOGLE_API_KEY=your_api_key_here
```

### 4. Prepare your resume
- Place your resume PDF in the project folder
- Update the file path in `rag_bot.py` line 29:
```python
loader = PyPDFLoader("your_resume.pdf")
```

## 🔧 Configuration

### Update API Keys
1. **Google API Key** (line 24):
   - Set environment variable `GOOGLE_API_KEY`

2. **Tavily API Key** (line 109):
   - Update the empty string with your Tavily API key:
   ```python
   tavily = TavilyClient(api_key="your_tavily_key_here")
   ```

### Customize LLM Parameters
- **Model**: Line 40 - Currently uses `gemini-2.5-flash`
- **Temperature**: Line 40 - Set to 0.3 (lower = more focused responses)
- **Chunk Size**: Line 33 - 1000 tokens per chunk
- **Chunk Overlap**: Line 33 - 200 tokens overlap

### Embedding Model
- Currently uses HuggingFace's `all-MiniLM-L6-v2` (lightweight, fast)
- Change line 34 to use different models if needed

## 📖 Usage

### Start the Application
```bash
python rag_bot.py
```

The application will:
1. Load and process your PDF resume
2. Create a vector database for similarity search
3. Start Flask server at `http://localhost:5000`

### Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

## 🌐 Web Interface

### **💬 Chat Tab**
- Ask questions about your resume
- Examples:
  - "What are my technical skills?"
  - "Tell me about my experience with machine learning"
  - "What projects have I worked on?"

### **📄 Modify Resume Tab**
1. Enter company name (e.g., "Google")
2. Paste the full job description
3. Click "Modify Resume"
4. Review the optimized content
5. Download as PDF

### **🔍 Company Projects Tab**
1. Enter company name (e.g., "Apple")
2. Click "Fetch Projects"
3. Get:
   - Project overviews
   - Tech stack analysis
   - Architecture patterns
   - Edge topics for interviews

## 🔌 API Endpoints

### POST `/chat`
Ask questions about resume content
```json
{
  "query": "What are my technical skills?"
}
```
**Response:**
```json
{
  "answer": "Based on your resume..."
}
```

### POST `/modify_resume`
Optimize resume for a job description
```json
{
  "company_name": "Google",
  "jd_text": "Full job description text..."
}
```
**Response:**
```json
{
  "modified_resume": "Optimized resume text..."
}
```

### GET `/download_resume`
Download the last generated modified resume
```
/download_resume?company=Google
```
**Returns:** PDF file download

### POST `/company_projects`
Research company projects and interview topics
```json
{
  "company_name": "Apple"
}
```
**Response:**
```json
{
  "projects": "Detailed company analysis..."
}
```

## 🏗️ Architecture

### Core Components

1. **Document Processing**
   - PDF loading and text extraction
   - Recursive text chunking with overlap
   - Vector embedding generation

2. **Vector Database**
   - FAISS for similarity search
   - Stores resume chunks as vectors
   - Retrieves top-3 most relevant chunks per query

3. **LLM Chain**
   - Gemini 2.5 Flash for generation
   - RAG chain combines retrieval + generation
   - Custom prompts for different tasks

4. **Web Server**
   - Flask application with modern dark UI
   - Real-time response streaming
   - PDF generation and download

5. **External APIs**
   - Google Generative AI
   - Tavily Search for company research
   - HuggingFace for embeddings

## 📝 How It Works

### RAG Pipeline (Chat & Resume Queries)
```
User Query → Vector Search → Retrieve Top-K Chunks → Gemini LLM → Response
```

### Resume Modification Pipeline
```
JD + Current Resume → Gemini Analysis → Keyword Integration → Optimized Resume → PDF Export
```

### Company Research Pipeline
```
Company Name → Tavily Web Search → Results Analysis → Gemini Interview Prep → Detailed Report
```

## 🎓 Example Use Cases

### Before an Interview
1. **Chat**: Ask about specific project details
2. **Company Projects**: Research the company's tech stack
3. **Resume Modifier**: Tailor your resume if applying

### During Preparation
1. Use the chat feature to quickly review your background
2. Study company edge topics from the Projects tab
3. Download tailored resume versions for each company

## ⚠️ Important Notes

- The LLM response is limited to information in your resume for chat queries
- Resume modification preserves original content - keywords are integrated naturally
- All modified resumes are saved with timestamps in the project folder
- Company research requires active internet connection (for Tavily API)

## 🛠️ Troubleshooting

### "GOOGLE_API_KEY not found"
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your_key"

# Windows CMD
set GOOGLE_API_KEY=your_key
```

### "No results found" on chat
- Ensure your PDF path is correct
- Verify the PDF is readable and contains text
- Check chunk size settings if PDF is very short

### PDF generation fails
- Ensure `reportlab` is installed
- Check folder permissions for RESUME_FOLDER
- Verify special characters in company name

## 📦 Output Files

Generated files are saved in `RESUME_FOLDER`:
```
C:\Users\LENOVO\Downloads\RAG\
├── Modified_Resume_CompanyName.pdf
├── Modified_Resume_AnotherCompany.pdf
└── ...
```

## 🔐 Security Notes

- Store API keys in environment variables, never in code
- Don't commit API keys to version control
- The application runs locally on `localhost:5000`
- All processing happens on your machine or via secure APIs

## 📚 Dependencies Summary

| Package | Purpose |
|---------|---------|
| `langchain` | LLM framework and chains |
| `langchain-google-genai` | Google Gemini integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `faiss-cpu` | Vector similarity search |
| `reportlab` | PDF generation |
| `flask` | Web server |
| `tavily-python` | Web search API |

## 🚀 Future Enhancements

- Multi-PDF resume support
- Resume version history tracking
- Interview question generation
- Real-time ATS score calculation
- Batch resume processing
- Export to multiple formats (DOCX, etc.)

## 📧 Support

For issues or improvements, refer to the `new.py` file for additional utilities or extend the existing functionality.

---

**Built with** ❤️ **for AI/ML engineering interview preparation**
