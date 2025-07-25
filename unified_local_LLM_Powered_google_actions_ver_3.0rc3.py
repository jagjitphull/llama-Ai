import os
import time
import pickle
import base64
import re
from pathlib import Path
from datetime import datetime, timedelta
import shutil
from collections import defaultdict
import numpy as np
import gradio as gr

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ----- Config -----
INDEX_DIR = "/quantum/Aillama/faiss_index/"
DOC_DIR = "/quantum/knowledge_base"
META_FILE = os.path.join(INDEX_DIR, "file_meta.pkl")
CLIENT_SECRETS = "client_secrets.json"

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']
CAL_SCOPES = ['https://www.googleapis.com/auth/calendar']
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

STATIC_DIR = "/home/ilg/ai_ai/ai_code/static"
#LOGO_URL = "static/gitss_logo.png"


QA_SYSTEM_INSTRUCTION = (
    "You are an expert assistant. Always answer in clear, well-formatted paragraphs, "
    "use line breaks for each new paragraph, and make lists or steps as separate lines.\n"
    "Avoid giving answers as a single long line."
)
QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        QA_SYSTEM_INSTRUCTION +
        "\n\nContext:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm_general = OllamaLLM(model="llama3", temperature=0.1)
llm_code = OllamaLLM(model="codellama", temperature=0.1)

category_descriptions = {}
category_embeddings = {}

# ---- Utility Functions ----
def format_answer(answer):
    answer = re.sub(r'\. (?=[A-Z])', '.\n', answer)
    answer = re.sub(r'(\d+\.) ', r'\n\1 ', answer)
    answer = re.sub(r'(- )', r'\n- ', answer)
    return answer.strip()

def get_gmail_service(send=False):
    scopes = GMAIL_SCOPES if send else [GMAIL_SCOPES[0]]
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, scopes)
    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)

def get_calendar_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, CAL_SCOPES)
    creds = flow.run_local_server(port=0)
    return build('calendar', 'v3', credentials=creds)

def get_drive_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, DRIVE_SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)

def import_gmail_emails(n=10):
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId='me', maxResults=n).execute()
        messages = results.get('messages', [])
        os.makedirs(f"{DOC_DIR}/gmail", exist_ok=True)
        for i, msg in enumerate(messages):
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = msg_data['payload']
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "(No Subject)")
            date = next((h['value'] for h in headers if h['name'] == 'Date'), "(No Date)")
            body = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
            elif 'body' in payload and 'data' in payload['body']:
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
            with open(f"{DOC_DIR}/gmail/email_{i+1}.txt", "w") as f:
                f.write(f"Subject: {subject}\nDate: {date}\n\n{body}")
        return f"Imported {len(messages)} emails to knowledge_base/gmail/"
    except Exception as e:
        return f"⚠️ Gmail import error: {e}"

def import_calendar_events(days=7):
    try:
        service = get_calendar_service()
        now = datetime.utcnow().isoformat() + 'Z'
        future = (datetime.utcnow() + timedelta(days=days)).isoformat() + 'Z'
        events_result = service.events().list(
            calendarId='primary', timeMin=now, timeMax=future, maxResults=20, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])
        os.makedirs(f"{DOC_DIR}/calendar", exist_ok=True)
        for i, event in enumerate(events):
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', '(No title)')
            with open(f"{DOC_DIR}/calendar/event_{i+1}.txt", "w") as f:
                f.write(f"{summary}\n{start}")
        return f"Imported {len(events)} events to knowledge_base/calendar/"
    except Exception as e:
        return f"⚠️ Calendar import error: {e}"

def import_gdrive_files(n=5):
    try:
        service = get_drive_service()
        results = service.files().list(
            pageSize=n, fields="files(id, name, mimeType)").execute()
        items = results.get('files', [])
        os.makedirs(f"{DOC_DIR}/gdrive", exist_ok=True)
        for item in items:
            fname = f"{DOC_DIR}/gdrive/{item['name']}.meta.txt"
            with open(fname, "w") as f:
                f.write(f"File: {item['name']}\nType: {item['mimeType']}\nID: {item['id']}")
        return f"Imported {len(items)} Google Drive file metadata to knowledge_base/gdrive/"
    except Exception as e:
        return f"⚠️ Drive import error: {e}"

def create_calendar_event(title, date, start_time, duration_minutes, attendees_emails):
    try:
        service = get_calendar_service()
        start_dt = f"{date}T{start_time}:00"
        fmt = "%Y-%m-%dT%H:%M:%S"
        dt_start = datetime.strptime(start_dt, fmt)
        dt_end = dt_start + timedelta(minutes=int(duration_minutes))
        end_dt = dt_end.strftime(fmt)
        attendees = [{'email': email.strip()} for email in attendees_emails.split(",") if email.strip()]
        event = {
            'summary': title,
            'start': {'dateTime': start_dt, 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': end_dt, 'timeZone': 'Asia/Kolkata'},
            'attendees': attendees
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"✅ Event created! Link: {event.get('htmlLink')}"
    except Exception as e:
        return f"⚠️ Calendar create error: {e}"

def send_gmail_email(to_email, subject, message_text):
    try:
        service = get_gmail_service(send=True)
        msg = MIMEText(message_text)
        msg['to'] = to_email
        msg['subject'] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        message = {'raw': raw}
        sent_msg = service.users().messages().send(userId="me", body=message).execute()
        return f"✅ Email sent! ID: {sent_msg['id']}"
    except Exception as e:
        return f"⚠️ Email send error: {e}"

def list_all_files(folder=DOC_DIR):
    files = []
    for file in Path(folder).glob("**/*"):
        if file.is_file() and file.suffix in (".pdf", ".epub", ".docx", ".txt"):
            stat = file.stat()
            rel_path = file.relative_to(folder)
            category = rel_path.parent.as_posix() if rel_path.parent != Path('.') else 'Uncategorized'
            files.append({
                "path": str(file),
                "mtime": stat.st_mtime,
                "category": category
            })
    return sorted(files, key=lambda x: x["path"])

def get_categories():
    categories = sorted(set(f["category"] for f in indexed_files))
    if "Uncategorized" in categories:
        categories.remove("Uncategorized")
        categories.insert(0, "Uncategorized")
    return ["All"] + categories

def get_category_descriptions(indexed_files):
    descs = defaultdict(list)
    for f in indexed_files:
        if f['category'] == 'Uncategorized': continue
        descs[f['category']].append(os.path.basename(f['path']))
    # Use first 3 file names as description for each category
    return {cat: " ".join(files[:3]) for cat, files in descs.items()}

def compute_category_embeddings(category_descriptions):
    cat_texts = [f"{cat}: {desc}" for cat, desc in category_descriptions.items()]
    if not cat_texts: return {}
    embeddings = embedding_model.embed_documents(cat_texts)
    return {cat: emb for cat, emb in zip(category_descriptions.keys(), embeddings)}

def file_table():
    files = indexed_files if indexed_files else []
    if not files:
        return "No documents indexed yet."
    table = "| File | Category | Last Modified |\n|------|----------|---------------|\n"
    for f in files:
        table += f"| {os.path.relpath(f['path'], DOC_DIR)} | {f['category']} | {time.ctime(f['mtime'])} |\n"
    return table

def get_file_fingerprint(files):
    return [(f['path'], f['mtime']) for f in files]

def save_file_meta(files):
    with open(META_FILE, "wb") as f:
        pickle.dump(get_file_fingerprint(files), f)

def load_file_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "rb") as f:
            return pickle.load(f)
    return None

def need_rebuild_index(files):
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return True
    prev = load_file_meta()
    now = get_file_fingerprint(files)
    return prev != now

def load_documents(files, category=None):
    docs = []
    for fileinfo in files:
        if category and fileinfo["category"] != category:
            continue
        file = Path(fileinfo["path"])
        try:
            if file.suffix == ".pdf":
                loaded = PyPDFLoader(str(file)).load()
            elif file.suffix == ".epub":
                loaded = UnstructuredEPubLoader(str(file)).load()
            elif file.suffix == ".docx":
                loaded = Docx2txtLoader(str(file)).load()
            elif file.suffix == ".txt":
                with open(str(file), encoding='utf-8', errors='ignore') as f:
                    from langchain_core.documents import Document
                    text = f.read()
                    loaded = [Document(page_content=text, metadata={"category": fileinfo["category"], "source": str(file)})]
            else:
                continue
            for doc in loaded:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["category"] = fileinfo["category"]
                doc.metadata["source"] = str(file)
            docs.extend(loaded)
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
    return docs

def build_and_save_index(files, embedding_model, category=None):
    print(f"🔄 Rebuilding FAISS index{' for category ' + category if category else ''}...")
    docs = load_documents(files, category)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents([d for d in docs if d.page_content.strip()])
    vector_db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_name = f"{category}_index" if category else "index"
    vector_db.save_local(os.path.join(INDEX_DIR, index_name))
    if not category:
        save_file_meta(files)
    print("✅ FAISS index rebuilt and saved.")
    return vector_db

def load_or_build_index(embedding_model, force=False, category=None):
    files = list_all_files()
    index_name = f"{category}_index" if category else "index"
    index_dir = os.path.join(INDEX_DIR, index_name)
    if category:
        cat_files = [f for f in files if f["category"] == category]
        if not cat_files:
            print(f"No files found for category '{category}', returning empty index.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            return FAISS.from_documents([], embedding_model), files
    if force or not os.path.exists(os.path.join(index_dir, "index.faiss")):
        return build_and_save_index(files, embedding_model, category), files
    else:
        print(f"🟢 Loading FAISS index from disk{' (category: ' + category + ')' if category else ''}...")
        vector_db = FAISS.load_local(
            index_dir, embedding_model, allow_dangerous_deserialization=True
        )
        return vector_db, files

# ------------- Main State -------------
vector_db, indexed_files = load_or_build_index(embedding_model)
category_descriptions = get_category_descriptions(indexed_files)
category_embeddings = compute_category_embeddings(category_descriptions)

def refresh_categories_and_embeddings():
    global category_descriptions, category_embeddings
    category_descriptions = get_category_descriptions(indexed_files)
    category_embeddings = compute_category_embeddings(category_descriptions)

def suggest_category_llm(query):
    if not query or not category_embeddings or not category_descriptions:
        return "All"
    query_emb = embedding_model.embed_query(query)
    best_cat = "All"
    best_score = -1
    for cat, emb in category_embeddings.items():
        score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat if best_score > 0.4 else "All"

def update_category_suggestion_llm(query):
    cat = suggest_category_llm(query)
    if cat and cat != "All":
        return gr.update(value=f"🔎 <b>Smart Suggestion:</b> <span style='color:#33f9c8'>{cat}</span>", visible=True), gr.update(visible=True)
    else:
        return gr.update(value="", visible=False), gr.update(visible=False)

def suggest_followups(answer):
    prompt = (
        "Given the following answer to a question about a document, "
        "suggest 3 to 5 specific and helpful follow-up questions a curious reader might ask. "
        "Return only the questions as a bullet list.\n\n"
        f"Answer: {answer}\n"
    )
    suggestions = llm_general.invoke(prompt)
    questions = []
    for line in suggestions.splitlines():
        line = line.strip()
        if line.startswith("-"):
            questions.append(line.lstrip("- ").strip())
        elif line and not questions:
            questions.append(line)
    return questions[:5] if questions else []

def pick_model(user_query, model_choice):
    if model_choice == "Llama 3":
        return llm_general, "llama3"
    if model_choice == "CodeLlama":
        return llm_code, "codellama"
    code_keywords = [
        "c code", "c program", "python", "function", "for loop", "while loop",
        "algorithm", "data structure", "write code", "source code",
        "snippet", "implement", "array", "pointer", "linked list",
        "fibonacci", "matrix", "sort", "file io", "recursion"
    ]
    q = user_query.lower()
    if any(k in q for k in code_keywords) or q.strip().startswith("write"):
        return llm_code, "codellama"
    return llm_general, "llama3"

def ask_llama(query, category, model_choice):
    use_category = None if (category == "All") else category
    global vector_db, qa, indexed_files
    vector_db, _ = load_or_build_index(embedding_model, category=use_category)
    llm, model_name = pick_model(query, model_choice)
    if hasattr(vector_db, 'index') and getattr(vector_db.index, 'ntotal', 0) == 0:
        return f"No documents found in the selected category. (Model: {model_name})", []
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT_TEMPLATE}
    )
    raw_answer = qa.invoke(query)
    if isinstance(raw_answer, dict):
        answer_text = raw_answer.get("result") or raw_answer.get("output") or ""
    else:
        answer_text = raw_answer or ""
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
    answer = format_answer(answer_text)
    suggestions = suggest_followups(answer)
    answer = f"**[{model_name}]**\n\n{answer}"
    return answer, suggestions

def ask_llama_and_reset(query, category, model_choice):
    answer_text, suggestions = ask_llama(query, category, model_choice)
    n_lines = max(12, min(50, answer_text.count('\n') + 2))
    return gr.update(value=answer_text, lines=n_lines), gr.update(choices=suggestions, value=None)

def reindex(dummy=None):
    global vector_db, qa, indexed_files
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db, indexed_files = load_or_build_index(embedding_model, force=True)
    qa = RetrievalQA.from_chain_type(
        llm=llm_general,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT_TEMPLATE}
    )
    # Refresh category suggestion vectors
    refresh_categories_and_embeddings()
    return gr.update(value="Index rebuilt successfully!"), file_table(), gr.update(choices=get_categories())

def upload_file(files, category):
    if not files:
        return gr.update(value="No file uploaded."), file_table(), gr.update(choices=get_categories())
    if not isinstance(files, list):
        files = [files]
    for file in files:
        filename = os.path.basename(file.name)
        target_dir = Path(DOC_DIR) / (category if category and category != "Uncategorized" else "")
        os.makedirs(target_dir, exist_ok=True)
        dst = target_dir / filename
        shutil.copy(file.name, dst)
        print(f"Uploaded {filename} to {dst}")
    out = reindex()
    refresh_categories_and_embeddings()
    return out

def update_query(selected_suggestion):
    return gr.update(value=selected_suggestion)

# ---------------- Gradio UI -----------------

with gr.Blocks(title="🦙 Smart AI Knowledge Center — Llama 3, CodeLlama, and Google Automation by GNU IT Solutions & Services") as demo:
    # ---- DARK MODE CSS ----
    gr.HTML("""
    <style>
        body, .gradio-container, .block, .gr-block, .tabs, .tabitem {
        background: #141a28 !important;
        color: #e6efff !important;
    }
    .gr-button, button, input[type="button"], .gr-text-input, .gr-dropdown, .gr-file, .gr-radio, .gr-number, .gr-markdown {
    background: #22304a !important;
    color: #fff !important;
    border: 1px solid #67a8d7 !important;
    }
    .gr-button, button {
    background: linear-gradient(90deg, #2862b6 0%, #22304a 100%) !important;
    }
    .gr-button:hover, button:hover {
    background: #265989 !important;
    }
    .gr-text-input, .gr-dropdown, .gr-radio, .gr-number, .gr-file {
    border-radius: 7px !important;
    }
    a, a:visited { color: #5dc7e0 !important; }
    ::-webkit-scrollbar { width: 10px; background: #22304a; }
    ::-webkit-scrollbar-thumb { background: #3578c1; border-radius: 8px; }
    </style>
    """)


    # LOGO AND BANNER
    gr.Markdown(f"""
    <div style="
        background: linear-gradient(90deg, #264178 0%, #65b7f3 100%);
        color: #fff;
        padding: 16px 20px 12px 20px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(38,65,120,0.16);
        margin-bottom: 16px;
        text-align: center;
        display:flex; align-items:center; flex-direction:column;">
        <img src="https://github.com/jagjitphull/assets_logo/blob/main/gitss_logo.png?raw=true " alt="GNU IT Solutions & Services Logo" style="height:60px;">
        <h1 style="font-size:2.3em; margin-bottom:0.2em;">🦙 Smart AI Knowledge Center</h1>
        <h2 style="font-weight:400; font-size:1.15em; color:#5dc7e0;">Llama 3, CodeLlama & Google Actions<br>
        <span style="font-size:0.97em;">by <b>GNU IT Solutions & Services</b></span></h2>
    </div>
    """)


    with gr.Tabs():
        # ---- Tab 1: Upload & Query ----
        with gr.TabItem("📂 Upload & Query"):
            gr.Markdown("""
            <div style="background: #23272c; border-left: 6px solid #4bd9c2; padding: 18px; margin-bottom:16px; border-radius:8px;">
            <b>Upload and organize PDFs, EPUBs, DOCX, TXT by category. Ask Llama 3 or CodeLlama. Rebuild index anytime.</b>
            </div>
            """)
            with gr.Row():
                upload = gr.File(label="Upload PDF, EPUB, DOCX files", file_types=['.pdf', '.epub', '.docx'], file_count="multiple")
                upload_category = gr.Textbox(label="Category/Subfolder", placeholder="(Optional, e.g., 'Business' or 'C Programming')")
                upload_btn = gr.Button("Upload and Index")
            with gr.Row():
                query = gr.Textbox(label="Ask a question about your docs (or Gmail, Calendar, Drive...)")
                category = gr.Dropdown(choices=get_categories(), label="Category", value="All", interactive=True)
                model_choice = gr.Dropdown(choices=["Auto", "Llama 3", "CodeLlama"], value="Auto", label="Model selection")
                ask_btn = gr.Button("Ask")
            with gr.Row():
                suggested_category = gr.HTML(label="Suggested Category", visible=False)
                use_suggested_btn = gr.Button("Use Suggested Category", visible=False)
            answer = gr.Textbox(
                label="Model's answer",
                lines=12,
                max_lines=50,
                show_copy_button=True
            )
            followups = gr.Radio(choices=[], label="Try a follow-up question:", interactive=True)
            reindex_btn = gr.Button("Reindex")
            reindex_status = gr.Textbox(label="Indexing status", value="")

            with gr.Accordion("📁 Show Indexed Files", open=False):
                files_box = gr.Markdown(file_table(), label="Indexed Files")


            upload_btn.click(fn=upload_file, inputs=[upload, upload_category], outputs=[reindex_status, files_box, category])
            ask_btn.click(fn=ask_llama_and_reset, inputs=[query, category, model_choice], outputs=[answer, followups])
            followups.change(fn=update_query, inputs=followups, outputs=query)
            reindex_btn.click(fn=reindex, outputs=[reindex_status, files_box, category])

            # LLM-powered suggestion
            query.change(fn=update_category_suggestion_llm, inputs=query, outputs=[suggested_category, use_suggested_btn])
            use_suggested_btn.click(fn=lambda q: gr.update(value=suggest_category_llm(q)), inputs=query, outputs=category)

        # ---- Tab 2: Import & Actions ----
        with gr.TabItem("📥 Import & Actions"):
            gr.Markdown("""
            <div style="background: #23272c; border-left: 6px solid #4bd9c2; padding: 18px; margin-bottom:16px; border-radius:8px;">
            <b>Integrate Gmail, Google Calendar, and Drive. Send email and schedule events. All local and private.</b>
            </div>
            """)
            with gr.Row():
                gmail_btn = gr.Button("Import Gmail Emails")
                cal_btn = gr.Button("Import Calendar Events")
                gdrive_btn = gr.Button("Import Google Drive Files")
                import_status = gr.Textbox(label="Import Status")
            gmail_btn.click(fn=import_gmail_emails, outputs=import_status)
            cal_btn.click(fn=import_calendar_events, outputs=import_status)
            gdrive_btn.click(fn=import_gdrive_files, outputs=import_status)

            gr.Markdown("## 📅 Schedule Google Calendar Event")
            evt_title = gr.Textbox(label="Event Title")
            evt_date = gr.Textbox(label="Date (YYYY-MM-DD)")
            evt_time = gr.Textbox(label="Start Time (HH:MM, 24hr)")
            evt_duration = gr.Number(label="Duration (minutes)", value=30)
            evt_emails = gr.Textbox(label="Attendees Emails (comma-separated)")
            evt_btn = gr.Button("Schedule Calendar Event")
            evt_status = gr.Textbox(label="Calendar Create Status")
            evt_btn.click(fn=create_calendar_event,
                          inputs=[evt_title, evt_date, evt_time, evt_duration, evt_emails],
                          outputs=evt_status)

            gr.Markdown("## 📧 Send Gmail Email")
            email_to = gr.Textbox(label="To Email")
            email_subject = gr.Textbox(label="Email Subject")
            email_body = gr.Textbox(label="Email Body", lines=4)
            email_btn = gr.Button("Send Email")
            email_send_status = gr.Textbox(label="Send Status")
            email_btn.click(fn=send_gmail_email,
                            inputs=[email_to, email_subject, email_body],
                            outputs=email_send_status)

    gr.Markdown(
        '<div style="text-align:center; color:#666; margin-top:24px; font-size:1em;">© 2025 GNU IT Solutions & Services. All rights reserved.</div>'
    )

if __name__ == "__main__":
    demo.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            auth=[("jp","ilg007")],
            #share=True,
            #allowed_paths=["static"]
            #debug=True
            )

