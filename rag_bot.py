import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from tavily import TavilyClient
from flask import Flask, request, jsonify, send_file
import threading

# ── bootstrap ──────────────────────────────────────────────────────────────
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

print("loading pdf..")
loader = PyPDFLoader("file_path.pdf")
docs = loader.load()

print("chunking doc content")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("creating vector db..")
vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

system_prompt = (
    "ur a helpful assistant answer if u provided content below else just ans no\n\n"
    "context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

RESUME_FOLDER = "C:\\Users\\LENOVO\\Downloads\\RAG"

# ── core functions ──────────────────────────────────────────────────────────
def modify_resume_for_jd(company_name, jd_text, docs, llm):
    resume_content = "\n".join([doc.page_content for doc in docs])
    messages = [
        SystemMessage(content="You are an expert resume optimizer and ATS specialist."),
        HumanMessage(content=f"""Company: {company_name}

Job Description:
{jd_text}

Current Resume:
{resume_content}

Task:
1. Extract all important keywords from the JD.
2. Identify where they naturally fit in the resume.
3. Rewrite bullet points to incorporate missing keywords WITHOUT fabricating skills.
4. Preserve all original sections and structure.
5. Output the complete modified resume as plain text only.""")
    ]
    response = llm.invoke(messages)
    return response.content


def save_resume_as_pdf(text, company_name):
    output_path = os.path.join(
        RESUME_FOLDER,
        f"Modified_Resume_{company_name.replace(' ', '_')}.pdf"
    )
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=15*mm, leftMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle('Heading', fontSize=12, fontName='Helvetica-Bold',
                                   textColor=colors.HexColor('#1a1a1a'), spaceAfter=4)
    body_style = ParagraphStyle('Body', fontSize=10, fontName='Helvetica',
                                textColor=colors.HexColor('#333333'), spaceAfter=3, leading=14)
    story = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.isupper() or line.endswith(":"):
            story.append(Paragraph(line, heading_style))
        else:
            line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(line, body_style))
    doc.build(story)
    return output_path


def get_company_projects(company_name, llm):
    tavily = TavilyClient(api_key="")
    results = tavily.search(
        query=f"{company_name} engineering projects tech stack methodology architecture 2024 2025",
        search_depth="advanced",
        max_results=10
    )
    context = "\n".join([r["content"] for r in results["results"]])
    messages = [
        SystemMessage(content="You are a senior tech interviewer and industry analyst helping a candidate deeply prepare for interviews."),
        HumanMessage(content=f"""Based on the search results below, give a deep interview-prep breakdown for {company_name}.

Search Results:
{context if context.strip() else "No detailed results found."}

For each notable project or product, provide:
1. Project/Product Name
2. What problem it solves
3. Tech stack and tools used
4. Architecture or system design highlights
5. Methodologies (Agile, ML pipelines, microservices, etc.)
6. Why it matters / business impact
7. Likely interview topics this project could lead to (DSA, system design, ML concepts)

End with a section called "EDGE TOPICS" — list 3-5 niche technical areas specific to {company_name} that most candidates won't know but will impress interviewers.

Be specific, technical, and dense. This is for a final-year AI/ML engineering student targeting a role at {company_name}.""")
    ]
    response = llm.invoke(messages)
    return response.content


# ── flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RAG Bot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a1a2e; padding: 16px 24px; border-bottom: 1px solid #2a2a4a; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; color: #a78bfa; }
  .tab-bar { display: flex; gap: 8px; padding: 12px 24px; background: #111; border-bottom: 1px solid #222; }
  .tab { padding: 8px 18px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; border: 1px solid #333; background: transparent; color: #aaa; transition: all 0.2s; }
  .tab.active { background: #a78bfa; color: #fff; border-color: #a78bfa; }
  .panel { display: none; flex: 1; overflow: hidden; }
  .panel.active { display: flex; flex-direction: column; }

  /* chat */
  #chat-messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 72%; padding: 10px 14px; border-radius: 12px; font-size: 0.92rem; line-height: 1.5; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #4f46e5; color: #fff; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: #1e1e2e; color: #e0e0e0; border-bottom-left-radius: 4px; }
  .msg.thinking { color: #888; font-style: italic; }
  .chat-input { display: flex; gap: 10px; padding: 16px 24px; border-top: 1px solid #222; background: #111; }
  .chat-input input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 0.92rem; outline: none; }
  .chat-input button { padding: 10px 20px; background: #a78bfa; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.92rem; }

  /* forms */
  .form-panel { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .form-panel label { font-size: 0.85rem; color: #aaa; margin-bottom: 4px; display: block; }
  .form-panel input, .form-panel textarea { width: 100%; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 0.92rem; outline: none; }
  .form-panel textarea { resize: vertical; min-height: 160px; }
  .form-panel button { padding: 11px 24px; background: #a78bfa; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; align-self: flex-start; }
  .result-box { background: #1e1e2e; border: 1px solid #2a2a4a; border-radius: 10px; padding: 16px; white-space: pre-wrap; font-size: 0.88rem; line-height: 1.6; max-height: 400px; overflow-y: auto; }
  .download-btn { padding: 9px 20px; background: #22c55e; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem; text-decoration: none; display: inline-block; margin-top: 8px; }
  .spinner { display: none; color: #a78bfa; font-style: italic; font-size: 0.88rem; }
  .spinner.show { display: block; }
</style>
</head>
<body>
<header>
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="#a78bfa" stroke-width="2"/><path d="M8 12h8M12 8v8" stroke="#a78bfa" stroke-width="2" stroke-linecap="round"/></svg>
  <h1>RAG Bot — Resume Intelligence</h1>
</header>

<div class="tab-bar">
  <button class="tab active" onclick="switchTab('chat')">💬 Chat</button>
  <button class="tab" onclick="switchTab('resume')">📄 Modify Resume</button>
  <button class="tab" onclick="switchTab('projects')">🔍 Company Projects</button>
</div>

<!-- CHAT -->
<div class="panel active" id="panel-chat">
  <div id="chat-messages"></div>
  <div class="chat-input">
    <input id="chat-input" placeholder="Ask anything about the resume..." onkeydown="if(event.key==='Enter') sendChat()"/>
    <button onclick="sendChat()">Send</button>
  </div>
</div>

<!-- RESUME MODIFIER -->
<div class="panel" id="panel-resume">
  <div class="form-panel">
    <div>
      <label>Company Name</label>
      <input id="res-company" placeholder="e.g. Razorpay"/>
    </div>
    <div>
      <label>Paste Job Description</label>
      <textarea id="res-jd" placeholder="Paste the full JD here..."></textarea>
    </div>
    <button onclick="modifyResume()">Modify Resume</button>
    <span class="spinner" id="res-spinner">⏳ Modifying resume, please wait...</span>
    <div id="res-result" style="display:none">
      <div class="result-box" id="res-text"></div>
      <a class="download-btn" id="res-download" href="#" download>⬇ Download PDF</a>
    </div>
  </div>
</div>

<!-- COMPANY PROJECTS -->
<div class="panel" id="panel-projects">
  <div class="form-panel">
    <div>
      <label>Company Name</label>
      <input id="proj-company" placeholder="e.g. Google"/>
    </div>
    <button onclick="fetchProjects()">Fetch Projects</button>
    <span class="spinner" id="proj-spinner">⏳ Fetching live data, please wait...</span>
    <div id="proj-result" style="display:none">
      <div class="result-box" id="proj-text"></div>
    </div>
  </div>
</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['chat','resume','projects'][i]===name));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
}

function addMsg(text, role) {
  const box = document.getElementById('chat-messages');
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
  return d;
}

async function sendChat() {
  const inp = document.getElementById('chat-input');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  addMsg(q, 'user');
  const thinking = addMsg('thinking...', 'bot thinking');
  const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query: q}) });
  const data = await res.json();
  thinking.className = 'msg bot';
  thinking.textContent = data.answer;
}

async function modifyResume() {
  const company = document.getElementById('res-company').value.trim();
  const jd = document.getElementById('res-jd').value.trim();
  if (!company || !jd) return alert('Fill in both fields.');
  document.getElementById('res-spinner').classList.add('show');
  document.getElementById('res-result').style.display = 'none';
  const res = await fetch('/modify_resume', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({company_name: company, jd_text: jd}) });
  const data = await res.json();
  document.getElementById('res-spinner').classList.remove('show');
  document.getElementById('res-text').textContent = data.modified_resume;
  document.getElementById('res-download').href = '/download_resume?company=' + encodeURIComponent(company);
  document.getElementById('res-result').style.display = 'block';
}

async function fetchProjects() {
  const company = document.getElementById('proj-company').value.trim();
  if (!company) return alert('Enter a company name.');
  document.getElementById('proj-spinner').classList.add('show');
  document.getElementById('proj-result').style.display = 'none';
  const res = await fetch('/company_projects', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({company_name: company}) });
  const data = await res.json();
  document.getElementById('proj-spinner').classList.remove('show');
  document.getElementById('proj-text').textContent = data.projects;
  document.getElementById('proj-result').style.display = 'block';
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "")
    response = rag_chain.invoke({"input": query})
    return jsonify({"answer": response["answer"]})


@app.route("/modify_resume", methods=["POST"])
def modify_resume():
    data = request.json
    company_name = data.get("company_name", "")
    jd_text = data.get("jd_text", "")
    modified = modify_resume_for_jd(company_name, jd_text, docs, llm)
    save_resume_as_pdf(modified, company_name)
    return jsonify({"modified_resume": modified})


@app.route("/download_resume")
def download_resume():
    company = request.args.get("company", "company")
    path = os.path.join(RESUME_FOLDER, f"Modified_Resume_{company.replace(' ', '_')}.pdf")
    return send_file(path, as_attachment=True)


@app.route("/company_projects", methods=["POST"])
def company_projects():
    company_name = request.json.get("company_name", "")
    projects = get_company_projects(company_name, llm)
    return jsonify({"projects": projects})


if __name__ == "__main__":
    print("RAG Bot ready → http://localhost:5000")
    app.run(debug=False, port=5000)