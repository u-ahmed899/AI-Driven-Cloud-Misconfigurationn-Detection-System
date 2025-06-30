from flask import Flask, redirect, request, jsonify, send_file, render_template, url_for
from flask_cors import CORS
import json
import os
import tempfile
import uuid
from datetime import datetime
import sqlite3
from fpdf import FPDF
from pathlib import Path
import re
import numpy as np
from fpdf.enums import XPos, YPos
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.json']
CORS(app)

# Load CNN model and tokenizer
try:
    model = load_model('json_cnn_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    MAX_SEQUENCE_LENGTH = 100
    print("CNN model and tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

# Initialize database
def init_db():
    conn = sqlite3.connect('uploads.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  Filename TEXT,
                  pdf_url TEXT,
                  Upload_Time TEXT,
                  Status TEXT,
                  Errors INTEGER,
                  Action TEXT,
                  cnn_prediction REAL)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_uploads_filename ON uploads(Filename)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_uploads_time ON uploads(Upload_Time)''')
    conn.commit()
    conn.close()

init_db()

class JSONValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.valid_json = None
        self.cnn_prediction = None
        self.cnn_confidence = None
        self.original_content = ""

    def validate(self, json_str):
        self.original_content = json_str
        
        try:
            self.valid_json = json.loads(json_str)
            syntax_valid = True
    
        except json.JSONDecodeError as e:
            self._parse_syntax_error(e, json_str)
            syntax_valid = False
        
        if model and tokenizer:
            self._cnn_analysis(json_str)
        
        return syntax_valid

    def _cnn_analysis(self, json_str):
        try:
            sequence = tokenizer.texts_to_sequences([json_str])
            padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
            
            prediction = model.predict(padded)[0][0]
            self.cnn_prediction = prediction
            self.cnn_confidence = abs(prediction - 0.5) * 2
            
            if prediction > 0.7:
                self.warnings.append({
                    'message': "CNN model detected potential structural issues",
                    'suggestion': "Review JSON structure carefully",
                    'error_type': "Structural Warning",
                    'confidence': f"{self.cnn_confidence*100:.1f}%"
                })
                
        except Exception as e:
            print(f"CNN analysis failed: {e}")
            self.warnings.append({
                'message': "Structural analysis unavailable",
                'suggestion': "Model processing error occurred",
                'error_type': "Analysis Warning"
            })

    def _parse_syntax_error(self, error, json_str):
        lines = json_str.split('\n')
        error_line = lines[error.lineno - 1] if error.lineno <= len(lines) else ""
        
        error_details = {
            'line': error.lineno,
            'column': error.colno,
            'message': error.msg,
            'context': error_line.strip(),
            'error_type': 'Syntax Error'
        }
        
        self.errors.append(error_details)

    def generate_report(self, filename):
        """Generate PDF report that works for both valid and invalid JSON"""
        pdf_path = os.path.join(tempfile.gettempdir(), f"report_{uuid.uuid4()}.pdf")
        
        class SafePDF(FPDF):
            def __init__(self):
                super().__init__()
                self.add_page()
                self.set_auto_page_break(auto=True, margin=15)
                self.set_margins(10, 10, 10)
                self.set_font("Helvetica", size=10)
            
            def add_error(self, title, details):
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(255, 0, 0)
                self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(0, 0, 0)
                self.set_font('Helvetica', '', 10)
                for key, value in details.items():
                    self.set_font('Helvetica', 'B', 10)
                    self.cell(40, 8, f"{key}:")
                    self.set_font('Helvetica', '', 10)
                    self.multi_cell(0, 8, str(value))
                    self.ln(1)
            
            def add_warning(self, title, details):
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(255, 165, 0)
                self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_text_color(0, 0, 0)
                self.set_font('Helvetica', '', 10)
                for key, value in details.items():
                    self.set_font('Helvetica', 'B', 10)
                    self.cell(40, 8, f"{key}:")
                    self.set_font('Helvetica', '', 10)
                    self.multi_cell(0, 8, str(value))
                    self.ln(1)

            def add_section(self, title, body):
                self.set_font('Helvetica', 'B', 12)
                self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_font('Helvetica', '', 10)
                self.multi_cell(0, 8, body)
                self.ln(5)
            
        pdf = SafePDF()
    
        # Add header
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(
            w=0,
            h=10,
            txt='JSON Validation Report',
            border=0,
            align='C',
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT
        )
        pdf.set_font("Helvetica", '', 12)
        pdf.cell(0, 8, f'File: {filename}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)
    
        # Add validation status
        status = "VALID" if not self.errors else f"INVALID ({len(self.errors)} errors)"
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(0, 128, 0) if not self.errors else pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f'Status: {status}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
    
        if not self.errors:
            pdf.add_section('Validation Result', 'JSON is valid!')
            pdf.add_section('Description:', 'The JSON is valid and can be used as is.')
                         
            if self.valid_json:
                pdf.add_section('Formatted JSON', json.dumps(self.valid_json, indent=2))
        else:
            pdf.add_section('Validation Result', 
                          f'Found {len(self.errors)} error(s) in JSON file')
            
            for error in self.errors:
                pdf.add_error(f"Error (Line {error['line']}, Column {error['column']})", {
                    'Type': error['error_type'],
                    'Message': error['message'],
                    'Context': error['context'],
                    'Suggestion': error.get('suggestion', 'Check JSON syntax')
                })
        
        if self.warnings:
            pdf.add_section('Formatting Warnings', 
                           f'Found {len(self.warnings)} potential formatting issue(s)')
            
            for warning in self.warnings:
                pdf.add_warning("Format Warning", {
                    'Message': warning['message'],
                    'Suggestion': warning['suggestion']
                })
        
        pdf.output(pdf_path)
        return pdf_path
    
def process_file(filepath_or_upload):
    try:
        if isinstance(filepath_or_upload, str):
            path = Path(filepath_or_upload)
            if not path.exists():
                return None, "File not found"
            if not path.suffix.lower() == '.json':
                return None, "Invalid file type. Only JSON files are allowed"
            
            filename = path.name
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        else:
            file = filepath_or_upload
            if file.filename == '':
                return None, "No selected file"
            
            if not file.filename.lower().endswith('.json'):
                return None, "Invalid file type. Only JSON files are allowed"
            
            filename = file.filename
            content = file.read().decode('utf-8')
        
        return content, filename, None
    
    except Exception as e:
        return None, None, str(e)

@app.route('/')
def home():
    return render_template('startup page.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            return redirect(url_for('upload_page'))
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/upload')
def upload_page():
    return render_template('upload file page.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/about')
def about_page():
    return render_template('about page.html')

@app.route('/admin')
def admin_page():
    return render_template('admin page.html')

@app.route('/api/validate', methods=['POST'])
def validate_json():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        
        if not file.filename.lower().endswith('.json'):
            return jsonify({"error": "Only JSON files allowed"}), 400

        try:
            content = file.read().decode('utf-8')
            file.seek(0)
            
            validator = JSONValidator()
            is_valid = validator.validate(content)
            
            pdf_path = validator.generate_report(file.filename)
            if not pdf_path or not os.path.exists(pdf_path):
                return jsonify({"error": "Failed to generate report"}), 500
                
            pdf_url = f"/api/download/{os.path.basename(pdf_path)}"
            
            # Store in database
            conn = sqlite3.connect('uploads.db')
            c = conn.cursor()
            c.execute("""INSERT INTO uploads 
                         (Filename, Upload_Time, pdf_url, Status, Errors, Action)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                     (file.filename, 
                      datetime.now().isoformat(), 
                      pdf_url, 
                      len(validator.errors), 
                      "Valid" if is_valid else "Invalid",
                      validator.cnn_prediction))
            conn.commit()
            conn.close()
            
            return jsonify({
                "status": "success" if is_valid else "validation_failed",
                "pdf_url": pdf_url,
                "filename": file.filename,
                "error_count": len(validator.errors),
                "warnings": validator.warnings,
                "cnn_prediction": validator.cnn_prediction,
                "cnn_confidence": validator.cnn_confidence
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/download/<filename>', methods=['GET'])
def download_pdf(filename):
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(pdf_path, as_attachment=True, mimetype='application/pdf')

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect('uploads.db')
        c = conn.cursor()
        c.execute("""SELECT Filename, Upload_Time, pdf_url, Status, Errors, Action
                     FROM uploads ORDER BY upload_time DESC""")
        rows = c.fetchall()
        history = [{
            'id': row[0],
            'filename': row[1],
            'Upload_Time': row[2],
            'Status': row[3],
            'Errors': row[4],
            'Action': row[5],
            'pdf_url': f"/api/download/report_{row[0]}.pdf"
        } for row in rows]
        conn.close()
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)