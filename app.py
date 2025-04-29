# app.py
import os
import shutil
import zipfile
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import ollama
import json
import gc
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
from langchain_core.runnables import RunnablePassthrough
import subprocess
import sys
from rapidocr_onnxruntime import RapidOCR
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_core.documents import Document
import fitz  # PyMuPDF

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_BRIEF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brief')
UPLOAD_REPORTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
CONTEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'context.txt')
TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_template.txt')
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports.db')
VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vectorstore')
BRIEF_VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brief_vectorstore')
ALLOWED_EXTENSIONS_BRIEF = {'pdf'}
ALLOWED_EXTENSIONS_REPORTS = {'zip'}

# Create directories if they don't exist
for directory in [UPLOAD_BRIEF_FOLDER, UPLOAD_REPORTS_FOLDER, VECTORSTORE_DIR, BRIEF_VECTORSTORE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        folder_name TEXT,
        upload_date TEXT,
        file_paths TEXT,
        feedback TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# Helper Functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Shared embedding and LLM
shared_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
shared_llm = Ollama(model="gemma3")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)

def get_image_embedding(img):
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features[0].cpu().numpy()

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        images.append(img_data)
    return images

def extract_images_and_ocr(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        images.append(img_data)
    ocr = RapidOCR()
    ocr_texts = []
    from PIL import Image
    import io
    for img_bytes in images:
        img = Image.open(io.BytesIO(img_bytes))
        result, _ = ocr(img)
        if result:
            ocr_texts.append('\n'.join([t[1] for t in result]))
    return ocr_texts

class BriefAgent:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    def process_brief(self, brief_path):
        print(f"[BriefAgent] Starting processing for: {brief_path}")
        chatbot_agent.log_status(f"Brief uploaded and processed: {os.path.basename(brief_path)}")
        print("[BriefAgent] Loading text and tables with UnstructuredPDFLoader...")
        loader = UnstructuredPDFLoader(brief_path, strategy="fast")
        documents = loader.load()
        print(f"[BriefAgent] Loaded {len(documents)} text/tables documents.")
        print("[BriefAgent] Extracting images and running OCR with PyMuPDF and RapidOCR...")
        ocr_texts = extract_images_and_ocr(brief_path)
        for idx, ocr_text in enumerate(ocr_texts):
            if ocr_text.strip():
                documents.append(Document(page_content=ocr_text, metadata={"source": brief_path, "ocr_image_index": idx}))
        print(f"[BriefAgent] Added {len(ocr_texts)} OCR image documents.")
        print("[BriefAgent] Splitting documents for vectorstore...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"[BriefAgent] Total split docs: {len(split_docs)}")
        print("[BriefAgent] Extracting images for multimodal LLM...")
        images = []
        doc = fitz.open(brief_path)
        for page in doc:
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append(img_data)
        print(f"[BriefAgent] Extracted {len(images)} images for multimodal LLM.")
        # Better cleanup of existing vectorstore
        if os.path.exists(BRIEF_VECTORSTORE_DIR):
            print("[BriefAgent] Cleaning up existing vectorstore...")
            try:
                vectorstore = Chroma(
                    persist_directory=BRIEF_VECTORSTORE_DIR,
                    embedding_function=self.embeddings
                )
                if hasattr(vectorstore, "_client"):
                    if hasattr(vectorstore._client, "close"):
                        vectorstore._client.close()
                    if hasattr(vectorstore._client, "_conn"):
                        if not vectorstore._client._conn.closed:
                            vectorstore._client._conn.close()
                try:
                    vectorstore.delete_collection()
                except:
                    pass
                del vectorstore
                gc.collect()
                time.sleep(1.0)
                temp_folder = BRIEF_VECTORSTORE_DIR + "_to_delete"
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder, ignore_errors=True)
                os.rename(BRIEF_VECTORSTORE_DIR, temp_folder)
                subprocess.Popen([sys.executable, "-c", f"import shutil; shutil.rmtree(r'{temp_folder}', ignore_errors=True)"])
            except Exception as e:
                print(f"Error cleaning up vectorstore: {str(e)}")
                try:
                    import sqlite3
                    sqlite3.connect(os.path.join(BRIEF_VECTORSTORE_DIR, "chroma.sqlite3")).close()
                except:
                    pass
                time.sleep(2.0)
                try:
                    shutil.rmtree(BRIEF_VECTORSTORE_DIR)
                except Exception as e2:
                    print(f"Failed to remove vectorstore directory: {str(e2)}")
        print("[BriefAgent] Creating new vectorstore...")
        os.makedirs(BRIEF_VECTORSTORE_DIR, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=BRIEF_VECTORSTORE_DIR
        )
        if hasattr(vectorstore, "_client") and hasattr(vectorstore._client, "close"):
            vectorstore._client.close()
        del vectorstore
        gc.collect()
        print("[BriefAgent] Generating context prompt with LLM...")
        if os.path.exists(TEMPLATE_FILE):
            with open(TEMPLATE_FILE, 'r') as f:
                template_content = f.read()
            prompt_template = ChatPromptTemplate.from_template(template_content)
        else:
            default_template = ("You are analyzing an assessment brief. "
                               "Extract the key rubrics, assessment criteria, and important details. "
                               "Look only for Coursework 2 components. "
                               "Create a context prompt that can be used to evaluate student reports against these criteria. "
                               "Make the prompt focused on helping an AI evaluate if reports meet the assessment requirements. "
                               "\n\nAssessment Brief: {brief_content}")
            with open(TEMPLATE_FILE, 'w') as f:
                f.write(default_template)
            prompt_template = ChatPromptTemplate.from_template(default_template)
        brief_content = "\n\n".join([doc.page_content for doc in documents])
        try:
            context_prompt = self.llm.invoke({"brief_content": brief_content, "images": images})
        except Exception:
            chain = prompt_template | self.llm | StrOutputParser()
            context_prompt = chain.invoke({"brief_content": brief_content})
        with open(CONTEXT_FILE, 'w') as f:
            f.write(context_prompt)
        print("[BriefAgent] Context prompt generated and saved.")
        return context_prompt

class ReportAgent:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    def process_report(self, report_path, folder_id):
        chatbot_agent.log_status(f"Report uploaded and processed: {os.path.basename(report_path)} (folder_id: {folder_id})")
        loader = PyPDFLoader(report_path)
        documents = loader.load()
        split_docs = self.text_splitter.split_documents(documents)
        for doc in split_docs:
            doc.metadata["folder_id"] = folder_id
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embeddings
        )
        vectorstore.add_documents(split_docs)
        vectorstore.persist()
        del vectorstore
        gc.collect()
        return len(split_docs)
    def analyze_report(self, folder_id):
        chatbot_agent.log_status(f"Report analyzed: {folder_id}")
        if not os.path.exists(CONTEXT_FILE):
            return "No assessment brief has been uploaded yet. Please upload a brief first."
        with open(CONTEXT_FILE, 'r') as f:
            context_prompt = f.read()
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=self.embeddings
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"folder_id": folder_id}}
        )
        prompt_template = ChatPromptTemplate.from_template(
            "You are analyzing student reports against assessment criteria. "
            "Based on the assessment context and the retrieved report content, provide detailed feedback. "
            "Focus on how well the report meets the assessment requirements. "
            "\n\nAssessment Context: {context}\n\nReport Content: {documents}"
        )
        # Use a simple chain: retrieve -> format prompt -> LLM -> parse
        def format_input(inputs):
            return {
                "context": inputs["context"],
                "documents": "\n\n".join([doc.page_content for doc in inputs["documents"]])
            }
        chain = (
            RunnablePassthrough()
            | (lambda x: {"context": context_prompt, "documents": retriever.get_relevant_documents(x)})
            | format_input
            | (lambda d: prompt_template.format(**d))
            | (lambda prompt: self.llm.invoke(prompt))
        )
        response = chain.invoke(folder_id)
        del vectorstore
        gc.collect()
        return response

class ChatbotAgent:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.status_log = []  # Store status messages
    def log_status(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.status_log.append(f"[{timestamp}] {message}")
        # Keep only the last 10 status messages
        self.status_log = self.status_log[-10:]
    def answer_question(self, question):
        brief_vectorstore = None
        reports_vectorstore = None
        status_msgs = list(self.status_log)  # Copy current status log
        try:
            current_vs_path = BRIEF_VECTORSTORE_DIR
            if os.path.exists(os.path.join(os.path.dirname(BRIEF_VECTORSTORE_DIR), "current_brief_vs.txt")):
                with open(os.path.join(os.path.dirname(BRIEF_VECTORSTORE_DIR), "current_brief_vs.txt"), "r") as f:
                    current_vs_path = f.read().strip()
            try:
                brief_vectorstore = Chroma(
                    persist_directory=current_vs_path,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"Error loading brief vectorstore: {str(e)}")
                brief_docs = []
            else:
                brief_docs = brief_vectorstore.similarity_search(question, k=3)
            try:
                reports_vectorstore = Chroma(
                    persist_directory=VECTORSTORE_DIR,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"Error loading reports vectorstore: {str(e)}")
                report_docs = []
            else:
                report_docs = reports_vectorstore.similarity_search(question, k=5)
            all_docs = brief_docs + report_docs
            if not all_docs:
                answer = "I couldn't find any relevant information to answer your question. Please try rephrasing or ask something related to the uploaded documents."
            else:
                prompt_template = ChatPromptTemplate.from_template(
                    "Answer the following question based on the provided documents. "
                    "If the answer is not in the documents, say so. "
                    "Question: {question}\n\nDocuments: {documents}"
                )
                prompt = prompt_template.format(question=question, documents="\n\n".join([doc.page_content for doc in all_docs]))
                try:
                    answer = self.llm.invoke(prompt)
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
            # Always include status log in the response
            status_section = "\n\n---\nStatus Log:\n" + "\n".join(status_msgs) if status_msgs else ""
            return str(answer) + status_section
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            status_section = "\n\n---\nStatus Log:\n" + "\n".join(status_msgs) if status_msgs else ""
            return f"I encountered an error while processing your question. Technical details: {str(e)}" + status_section
        finally:
            if brief_vectorstore is not None:
                try:
                    if hasattr(brief_vectorstore, "_client") and hasattr(brief_vectorstore._client, "close"):
                        brief_vectorstore._client.close()
                    if hasattr(brief_vectorstore, "_client") and hasattr(brief_vectorstore._client, "_conn"):
                        if not brief_vectorstore._client._conn.closed:
                            brief_vectorstore._client._conn.close()
                except Exception as e:
                    print(f"Error closing brief vectorstore: {str(e)}")
            if reports_vectorstore is not None:
                try:
                    if hasattr(reports_vectorstore, "_client") and hasattr(reports_vectorstore._client, "close"):
                        reports_vectorstore._client.close()
                    if hasattr(reports_vectorstore, "_client") and hasattr(reports_vectorstore._client, "_conn"):
                        if not reports_vectorstore._client._conn.closed:
                            reports_vectorstore._client._conn.close()
                except Exception as e:
                    print(f"Error closing reports vectorstore: {str(e)}")
            gc.collect()

# Initialize agents with shared embedding and LLM
brief_agent = BriefAgent(shared_embeddings, shared_llm)
report_agent = ReportAgent(shared_embeddings, shared_llm)
chatbot_agent = ChatbotAgent(shared_embeddings, shared_llm)

# Routes
@app.route('/', methods=['GET'])
def index():
    # Get all reports from database
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM reports ORDER BY upload_date DESC")
    reports = [dict(row) for row in c.fetchall()]
    conn.close()
    # For each report, parse file_paths from JSON
    for report in reports:
        report['files'] = json.loads(report['file_paths'])
    # Find uploaded brief (if any)
    brief_files = [f for f in os.listdir(UPLOAD_BRIEF_FOLDER) if f.lower().endswith('.pdf')]
    brief_filename = brief_files[0] if brief_files else None
    return render_template('index.html', reports=reports, brief_filename=brief_filename)

@app.route('/upload_brief', methods=['POST'])
def upload_brief():
    if 'brief' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['brief']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_BRIEF):
        # Clear brief directory
        for f in os.listdir(UPLOAD_BRIEF_FOLDER):
            os.remove(os.path.join(UPLOAD_BRIEF_FOLDER, f))
        
        # Save new brief
        filename = secure_filename(file.filename)
        brief_path = os.path.join(UPLOAD_BRIEF_FOLDER, filename)
        file.save(brief_path)
        
        # Process brief using agent
        try:
            context_prompt = brief_agent.process_brief(brief_path)
            flash(f'Brief uploaded and processed successfully. Context prompt generated.')
        except Exception as e:
            flash(f'Error processing brief: {str(e)}')
    
    return redirect(url_for('index'))

@app.route('/upload_reports', methods=['POST'])
def upload_reports():
    if 'reports' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['reports']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_REPORTS):
        zip_path = os.path.join(UPLOAD_REPORTS_FOLDER, secure_filename(file.filename))
        file.save(zip_path)
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
        # Extract zip to a temp folder
        temp_extract_dir = os.path.join(UPLOAD_REPORTS_FOLDER, f"tmp_{uuid.uuid4()}")
        os.makedirs(temp_extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        os.remove(zip_path)
        # For each top-level folder, create a report entry
        top_level_dirs = [d for d in os.listdir(temp_extract_dir) if os.path.isdir(os.path.join(temp_extract_dir, d))]
        if not top_level_dirs:
            # If no folders, treat all PDFs in root as one entry
            top_level_dirs = ['.']
        for folder in top_level_dirs:
            if folder == '.':
                folder_path = temp_extract_dir
                folder_name = os.path.splitext(secure_filename(file.filename))[0]
            else:
                folder_path = os.path.join(temp_extract_dir, folder)
                folder_name = folder
            folder_id = str(uuid.uuid4())
            os.makedirs(os.path.join(UPLOAD_REPORTS_FOLDER, folder_id), exist_ok=True)
            file_paths = []
            pdf_count = 0
            for root, dirs, files in os.walk(folder_path):
                for pdf_file in files:
                    if pdf_file.lower().endswith('.pdf'):
                        src_pdf_path = os.path.join(root, pdf_file)
                        dest_pdf_path = os.path.join(UPLOAD_REPORTS_FOLDER, folder_id, pdf_file)
                        shutil.copy2(src_pdf_path, dest_pdf_path)
                        relative_path = os.path.relpath(dest_pdf_path, UPLOAD_REPORTS_FOLDER)
                        file_paths.append({'name': pdf_file, 'path': relative_path})
                        try:
                            report_agent.process_report(dest_pdf_path, folder_id)
                            pdf_count += 1
                        except Exception as e:
                            flash(f'Error processing {pdf_file}: {str(e)}')
            # Save to database
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute(
                "INSERT INTO reports (id, folder_name, upload_date, file_paths, feedback) VALUES (?, ?, ?, ?, ?)",
                (folder_id, folder_name, datetime.now().isoformat(), json.dumps(file_paths), '')
            )
            conn.commit()
            conn.close()
            flash(f'Successfully uploaded and processed {pdf_count} PDFs in folder {folder_name}')
        shutil.rmtree(temp_extract_dir)
    return redirect(url_for('index'))

@app.route('/delete_report/<report_id>', methods=['POST'])
def delete_report(report_id):
    # Get report info
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    report = c.fetchone()
    
    if report:
        # Delete files
        folder_path = os.path.join(UPLOAD_REPORTS_FOLDER, report_id)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        
        # Delete from database
        c.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        
        flash('Report deleted successfully')
    else:
        flash('Report not found')
    
    conn.close()
    return redirect(url_for('index'))

@app.route('/analyze_report/<report_id>', methods=['POST'])
def analyze_report(report_id):
    try:
        feedback = report_agent.analyze_report(report_id)
        
        # Update database
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE reports SET feedback = ? WHERE id = ?", (feedback, report_id))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'feedback': feedback})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    answer = chatbot_agent.answer_question(question)
    return jsonify({'status': 'success', 'answer': answer})

@app.route('/get_template', methods=['GET'])
def get_template():
    try:
        if os.path.exists(TEMPLATE_FILE):
            with open(TEMPLATE_FILE, 'r') as f:
                template_content = f.read()
        else:
            # Create default template"
            template_content = "You are analyzing an assessment brief. Extract the key rubrics, assessment criteria, and important details. Look only for Final Portfolio components in Section 6 - Plans, requirements, issues, and implementation. DO NOT include any material from Section 7 General Guidance. Assessment Brief: {brief_content}. Create a context prompt that can be used to evaluate student reports against these assessment criteria from the table in Section 6. Make the prompt focused on helping an AI evaluate if reports meet the assessment requirements, with one paragraph each for the criteria provided (Plans, requirements, issues, and implementation), Do not include any questions as the answer is a final statement. Do not include any AI preamble as well."
            with open(TEMPLATE_FILE, 'w') as f:
                f.write(template_content)
        
        return jsonify({'template_content': template_content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_template', methods=['POST'])
def save_template():
    try:
        data = request.get_json()
        template_content = data.get('template_content', '')
        
        # Make sure the template contains the required placeholder
        if '{brief_content}' not in template_content:
            return jsonify({'error': 'Template must contain {brief_content} placeholder'}), 400
        
        with open(TEMPLATE_FILE, 'w') as f:
            f.write(template_content)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_context', methods=['GET'])
def get_context():
    try:
        if os.path.exists(CONTEXT_FILE):
            with open(CONTEXT_FILE, 'r') as f:
                context_content = f.read()
        else:
            context_content = ''
        return jsonify({'context_content': context_content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_context', methods=['POST'])
def save_context():
    try:
        data = request.get_json()
        context_content = data.get('context_content', '')
        with open(CONTEXT_FILE, 'w') as f:
            f.write(context_content)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_REPORTS_FOLDER, filename)

@app.route('/brief/<filename>')
def download_brief(filename):
    return send_from_directory(UPLOAD_BRIEF_FOLDER, filename)

@app.route('/reset_all', methods=['POST'])
def reset_all():
    try:
        # Remove database file
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        # Move locked folders to temp and delete in subprocess
        for folder in [UPLOAD_BRIEF_FOLDER, UPLOAD_REPORTS_FOLDER, VECTORSTORE_DIR, BRIEF_VECTORSTORE_DIR]:
            if os.path.exists(folder):
                temp_folder = folder + "_to_delete"
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder, ignore_errors=True)
                os.rename(folder, temp_folder)
                # Spawn a subprocess to delete the folder
                subprocess.Popen([sys.executable, "-c", f"import shutil; shutil.rmtree(r'{temp_folder}', ignore_errors=True)"])
                os.makedirs(folder, exist_ok=True)
        # Re-initialize database
        init_db()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)