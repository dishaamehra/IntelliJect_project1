#main.py
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
# from sqlalchemy import text
from sqlalchemy import text # Use alias to avoid conflicts
from database import engine, SessionLocal, PYQ
from utils import extract_text_from_pdf
from rag_pipeline import get_relevant_pyqs
import tempfile
import datetime
import fitz  # PyMuPDF
from database import get_session
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from nltk.tokenize import sent_tokenize
from langchain.chat_models import ChatOpenAI
from study_planner import StudyPlanGenerator, create_study_plan
from  enhanced_smart_database_builder import generate_question_hash,check_question_exists_by_hash,save_questions_to_database_with_hash
import cv2
#from supabase_auth import init_auth, require_auth, get_current_user_email, get_current_user_id
from auth_manager import init_auth, require_auth, get_current_user_email, get_current_user_id
import numpy as np
import hashlib
import tempfile
import json
import re
from typing import Dict, List, Optional, Tuple
import pytesseract
import os
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rag_pipeline import (
    get_relevant_pyqs_with_filters, 
    get_database_filter_options, 
    validate_filters,
    get_filter_statistics
)
import math
import random
import base64

load_dotenv()

# University Configuration
UNIVERSITY_CONFIG = {
    "NSUT": {
        "branches": ["CSE", "CSE-AI", "CSE-DS", "EE", "ECE", "ECE-VLSI Design", "IT", "IT-Network and Information Security", "ICE", "M&C", "ME"],
        "subjects": {
            "CSE": ["Data Structures", "Computer Networks", "Database Management", "Operating Systems", "Software Engineering"],
            "CSE-AI": ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Computer Vision", "Natural Language Processing"],
            "CSE-DS": ["Data Science", "Big Data Analytics", "Data Mining", "Statistics", "Python Programming"],
            "EE": ["Circuit Analysis", "Power Systems", "Control Systems", "Electronics", "Electromagnetic Theory"],
            "ECE": ["Digital Electronics", "Communication Systems", "Signal Processing", "Microprocessors", "VLSI Design"],
            "ECE-VLSI Design": ["VLSI Design", "Digital IC Design", "Analog IC Design", "System on Chip", "Verification"],
            "IT": ["Information Systems", "Web Technologies", "Database Systems", "Network Security", "Cloud Computing"],
            "IT-Network and Information Security": ["Network Security", "Cryptography", "Ethical Hacking", "Information Security", "Cyber Security"],
            "ICE": ["Instrumentation", "Control Engineering", "Process Control", "Sensors", "Automation"],
            "M&C": ["Manufacturing Processes", "CAD/CAM", "Production Engineering", "Quality Control", "Industrial Engineering"],
            "ME": ["Thermodynamics", "Fluid Mechanics", "Machine Design", "Manufacturing", "Heat Transfer"]
        }
    },
    "DTU": {
        "branches": ["CSE", "CSE-DS and Analytics", "EE", "ECE", "ECE-VLSI Design", "IT", "IT-Cyber Security", "Bio-Technology", "Software Engineering", "Civil Engineering", "ME"],
        "subjects": {
            "CSE": ["Data Structures", "Algorithms", "Computer Networks", "Database Systems", "Operating Systems"],
            "CSE-DS and Analytics": ["Data Analytics", "Machine Learning", "Big Data", "Statistical Methods", "Data Visualization"],
            "EE": ["Electrical Circuits", "Power Electronics", "Control Systems", "Electrical Machines", "Power Systems"],
            "ECE": ["Electronic Circuits", "Digital Signal Processing", "Communication Engineering", "Microwave Engineering", "Embedded Systems"],
            "ECE-VLSI Design": ["VLSI Technology", "Digital System Design", "Analog Circuit Design", "CMOS Design", "Physical Design"],
            "IT": ["Information Technology", "Web Development", "Database Management", "System Administration", "Network Management"],
            "IT-Cyber Security": ["Cyber Security", "Network Security", "Information Security", "Digital Forensics", "Security Analytics"],
            "Bio-Technology": ["Biochemistry", "Molecular Biology", "Genetics", "Bioprocess Engineering", "Bioinformatics"],
            "Software Engineering": ["Software Design", "Software Testing", "Project Management", "Requirements Engineering", "Agile Methodology"],
            "Civil Engineering": ["Structural Engineering", "Environmental Engineering", "Transportation", "Geotechnical Engineering", "Construction Management"],
            "ME": ["Mechanical Design", "Manufacturing Engineering", "Thermal Engineering", "Industrial Engineering", "Robotics"]
        }
    },
    "IGDTUW": {
        "branches": ["CSE", "CSE-AI", "IT", "ECE", "EE", "Applied Mathematics"],
        "subjects": {
            "CSE": {
                "Semester 1": ["Probability and Statistics", "Environmental Sciences", "Programming with Python", 
                              "CAD Modelling", "Cyber Security Awareness", "Applied Mechanics", 
                              "Web Application Development", "Basics of Electrical and Electronics Engineering", 
                              "IT Workshop", "Communication Skills"],
                "Semester 2": ["Applied Mathematics", "Applied Physics", "Data Structures", "CAD Modelling", 
                              "Cyber Security Awareness", "Applied Mechanics", "Web Application Development", 
                              "Basics of Electrical and Electronics Engineering", "Introduction to Data Science", 
                              "Soft Skills and Personality Development"],
                "Other": ["Programming Fundamentals", "Algorithms", "Computer Networks", "Database Systems"]
            },
            "CSE-AI": {
                "Semester 1": ["Probability and Statistics", "Environmental Sciences", "Programming with Python", 
                              "CAD Modelling", "IT Workshop", "Applied Mechanics", 
                              "Basics of Electrical and Electronics Engineering", "Web Application Development", 
                              "Communication Skills"],
                "Semester 2": ["Applied Mathematics", "Applied Physics", "Object Oriented Programming", 
                              "CAD Modelling", "IT Workshop", "Applied Mechanics", 
                              "Basics of Electrical and Electronics Engineering", "Introduction to Data Science", 
                              "Soft Skills and Personality Development","Cyber Security"],
                "Semester 3": ["Artificial Intelligence","Discrete Mathematics", "Object Oriented Programming", "Operation Management", "Database Management System"],
                "Other": [ "Machine Learning", "Deep Learning", "Computer Vision", "Natural Language Processing"]
            },
            "IT": ["Information Systems", "Web Technologies", "Software Engineering", "Network Administration", "IT Project Management"],
            "ECE": ["Electronic Devices", "Communication Systems", "Digital Electronics", "Microprocessors", "Control Systems"],
            "EE": ["Electrical Engineering", "Power Systems", "Control Theory", "Electronics", "Circuit Analysis"],
            "Applied Mathematics": ["Calculus", "Linear Algebra", "Probability and Statistics", "Discrete Mathematics", "Numerical Methods"]
        }
    },
    "VIT University": {
        "branches": ["CSE-AI", "CSE", "IT", "AI-ML", "ECE", "ECE-AI", "MAE"],
        "subjects": {
            "CSE-AI": ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Computer Vision", "Robotics"],
            "CSE": ["Programming", "Data Structures", "Algorithms", "Computer Networks", "Software Engineering"],
            "IT": ["Information Technology", "Database Systems", "Web Technologies", "System Analysis", "IT Infrastructure"],
            "AI-ML": ["Machine Learning", "Artificial Intelligence", "Data Science", "Neural Networks", "Pattern Recognition"],
            "ECE": ["Electronics", "Communication Systems", "Signal Processing", "Digital Electronics", "Microprocessors"],
            "ECE-AI": ["AI in Electronics", "Smart Systems", "IoT", "Embedded AI", "Intelligent Control"],
            "MAE": ["Mechanical Engineering", "Aerospace Engineering", "Manufacturing", "Design Engineering", "Thermal Systems"]
        }
    }
}

# Configure Tesseract path from environment variable
tesseract_path = os.getenv("TESSERACT_CMD")
if tesseract_path and os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Fallback to default Windows installation path
    default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(default_path):
        pytesseract.pytesseract.tesseract_cmd = default_path
    else:
        print("Warning: Tesseract not found. OCR features will not work.")


# Smart Database Builder utility functions
class SmartDatabaseBuilder:
    def __init__(self):
        pass
    
    def generate_question_hash(self, question: str) -> str:
        """Generate a hash for question deduplication"""
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def preprocess_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        try:
            # Convert PIL Image to OpenCV format
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        except Exception as e:
            st.warning(f"Image preprocessing failed: {e}")
            return np.array(image) if isinstance(image, Image.Image) else image
    
    def extract_text_from_image(self, image) -> str:
        """Extract text from image using OCR"""
        try:
            # Convert to PIL Image if needed
            if hasattr(image, 'read'):
                image = Image.open(image)
            
            # Preprocess image
            processed = self.preprocess_image_for_ocr(image)
            processed_pil = Image.fromarray(processed)
            
            # OCR with custom config
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,?!()[]{}:;"\'-+=/\n '
            text = pytesseract.image_to_string(processed_pil, config=custom_config)
            
            return text.strip()
        except Exception as e:
            st.error(f"OCR processing failed: {e}")
            return ""
    
    def extract_text_from_pdf_enhanced(self, pdf_file) -> str:
        """Enhanced PDF text extraction with OCR fallback"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            doc = fitz.open(tmp_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try text extraction first
                text = page.get_text()
                
                if len(text.strip()) < 50:  # If little text found, use OCR
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # Higher resolution
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # OCR the image
                    image = Image.open(BytesIO(img_data))
                    ocr_text = self.extract_text_from_image(image)
                    full_text += ocr_text + "\n"
                else:
                    full_text += text + "\n"
            
            doc.close()
            import os
            os.unlink(tmp_path)
            
            return full_text.strip()
        
        except Exception as e:
            st.error(f"PDF processing failed: {e}")
            return ""
    
    def parse_questions_with_llm(self, text: str, context: Dict) -> List[Dict]:
        """Use LLM to extract and parse questions from text"""
        try:
            llm = ChatOpenAI(temperature=0, model="gpt-4")
            
            prompt = f"""
            Extract individual questions from the following exam paper text. For each question, identify:
            1. The complete question text
            2. Question number (if available)
            3. Marks allocated (if mentioned)
            4. Sub-topic/concept being tested (infer from question content)
            5. Unit (if mentioned or can be inferred)
            
            Context:
            - Subject: {context.get('subject', 'Unknown')}
            - Branch: {context.get('branch', 'Unknown')}
            - Year: {context.get('year', 'Unknown')}
            - Semester: {context.get('semester', 'Unknown')}
            
            Text to analyze:
            {text}
            
            Return the result as a JSON array where each question is an object with these fields:
            - question: (complete question text)
            - marks: (numeric value, use 1 if not specified)
            - sub_topic: (inferred topic/concept)
            - unit: (unit number if identifiable)
            - question_number: (question number if available)
            
            Only include actual questions, not instructions or headers.
            """
            
            response = llm.predict(prompt)
            
            # Try to parse JSON response
            try:
                # Clean the response to extract JSON
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    questions = json.loads(json_str)
                    return questions
            except json.JSONDecodeError:
                st.warning("LLM returned invalid JSON, attempting to parse manually")
                return []
                
        except Exception as e:
            st.error(f"Error in LLM processing: {e}")
            return []
    
    def validate_question_data(self, question_data: Dict) -> Tuple[bool, str]:
        """Validate extracted question data"""
        if not question_data.get('question'):
            return False, "Question text is missing"
        
        if len(question_data['question'].strip()) < 10:
            return False, "Question text is too short"
        
        # Check if it looks like a valid question
        question_text = question_data['question'].lower()
        question_indicators = ['what', 'how', 'why', 'explain', 'define', 'solve', 'find', 'prove', 'derive', 'calculate', 'evaluate', 'determine', 'show', 'verify', 'trace', 'compute']
        
        if not any(indicator in question_text for indicator in question_indicators):
            return False, "Text doesn't appear to be a question"
        
        # Avoid headers and instructions
        avoid_terms = ['time:', 'marks:', 'instructions:', 'note:', 'attempt']
        if any(term in question_text for term in avoid_terms):
            return False, "Appears to be instruction or header text"
        
        return True, "Valid question"
    
    def save_questions_to_database(self, questions: List[Dict], context: Dict) -> Tuple[int, int, int]:
        """Save questions to database, return (added, updated, skipped) counts"""
        added_count = 0
        updated_count = 0
        skipped_count = 0
        
        try:
            with SessionLocal() as db:
                for q_data in questions:
                    try:
                        # Generate hash for deduplication
                        question_hash = self.generate_question_hash(q_data['question'])
                        
                        # Check if question already exists
                        existing = db.query(PYQ).filter(PYQ.question_hash == question_hash).first()
                        
                        if existing:
                            # Update frequency
                            existing.frequency = getattr(existing, 'frequency', 1) + 1
                            # Update year if not already included
                            if hasattr(existing, 'year') and context['year'] not in str(existing.year):
                                existing.year = f"{existing.year},{context['year']}"
                            updated_count += 1
                        else:
                            # Create new question with enhanced fields
                            new_question = PYQ(
                                question=q_data['question'],
                                subject=context['subject'],
                                sub_topic=q_data.get('sub_topic', 'General'),
                                year=context['year'],
                                marks=float(q_data.get('marks', 1))
                            )
                            
                            # Add new fields if they exist in the model
                            if hasattr(PYQ, 'question_hash'):
                                new_question.question_hash = question_hash
                            if hasattr(PYQ, 'unit'):
                                new_question.unit = q_data.get('unit', '')
                            if hasattr(PYQ, 'semester'):
                                new_question.semester = context.get('semester', '')
                            if hasattr(PYQ, 'branch'):
                                new_question.branch = context.get('branch', '')
                            if hasattr(PYQ, 'college'):
                                new_question.college = context.get('college', '')
                            if hasattr(PYQ, 'course'):
                                new_question.course = context.get('course', '')
                            if hasattr(PYQ, 'frequency'):
                                new_question.frequency = 1
                            
                            db.add(new_question)
                            added_count += 1
                    
                    except Exception as e:
                        st.warning(f"Error processing question: {e}")
                        skipped_count += 1
                        continue
                
                db.commit()
                
        except Exception as e:
            st.error(f"Database error: {e}")
            return 0, 0, len(questions)
        
        return added_count, updated_count, skipped_count

# Initialize managers
study_planner = StudyPlanGenerator()
smart_db = SmartDatabaseBuilder()

# Utility functions
def get_available_branches(university):
    """Get available branches for selected university"""
    if university == "All Colleges" or university not in UNIVERSITY_CONFIG:
        # Return all branches from all universities
        all_branches = set()
        for config in UNIVERSITY_CONFIG.values():
            all_branches.update(config["branches"])
        return ["All Branches"] + sorted(list(all_branches))
    return ["All Branches"] + UNIVERSITY_CONFIG[university]["branches"]

def get_available_subjects(university, branch, semester=None):
    """Get available subjects for selected university, branch, and optionally semester"""
    if university == "All Colleges" or university not in UNIVERSITY_CONFIG:
        # Return all subjects from all universities
        all_subjects = set()
        for config in UNIVERSITY_CONFIG.values():
            for branch_subjects in config["subjects"].values():
                if isinstance(branch_subjects, dict):  # Semester-wise subjects
                    for sem_subjects in branch_subjects.values():
                        if isinstance(sem_subjects, list):
                            all_subjects.update(sem_subjects)
                        else:
                            all_subjects.add(sem_subjects)
                elif isinstance(branch_subjects, list):  # Direct list of subjects
                    all_subjects.update(branch_subjects)
        return sorted(list(all_subjects))
    
    if branch == "All Branches":
        # Return all subjects for this university
        all_subjects = set()
        for branch_subjects in UNIVERSITY_CONFIG[university]["subjects"].values():
            if isinstance(branch_subjects, dict):  # Semester-wise subjects
                for sem_subjects in branch_subjects.values():
                    if isinstance(sem_subjects, list):
                        all_subjects.update(sem_subjects)
            elif isinstance(branch_subjects, list):  # Direct list of subjects
                all_subjects.update(branch_subjects)
        return sorted(list(all_subjects))
    
    if branch in UNIVERSITY_CONFIG[university]["subjects"]:
        branch_subjects = UNIVERSITY_CONFIG[university]["subjects"][branch]
        
        # Check if subjects are organized by semester
        if isinstance(branch_subjects, dict):
            if semester and semester in branch_subjects:
                return branch_subjects[semester]
            else:
                # Return all subjects from all semesters for this branch
                all_subjects = set()
                for sem_subjects in branch_subjects.values():
                    if isinstance(sem_subjects, list):
                        all_subjects.update(sem_subjects)
                return sorted(list(all_subjects))
        elif isinstance(branch_subjects, list):
            return branch_subjects
    
    return []

def get_available_semesters(university, branch):
    """Get available semesters for selected university and branch"""
    if (university == "All Colleges" or university not in UNIVERSITY_CONFIG or 
        branch == "All Branches" or branch not in UNIVERSITY_CONFIG[university]["subjects"]):
        return []
    
    branch_subjects = UNIVERSITY_CONFIG[university]["subjects"][branch]
    if isinstance(branch_subjects, dict):
        return list(branch_subjects.keys())
    
    return []

def extract_answer_from_chunk(chunk, question):
    prompt = f"""
Given the following notes and a question, extract the exact sentence(s) from the notes that directly answer the question if possible. Only return the excerpt(s), not any explanation.

Notes:
\"\"\"{chunk}\"\"\"

Question:
\"\"\"{question}\"\"\"

Answer/excerpt:
"""
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        answer = llm.predict(prompt).strip()
        return answer
    except Exception as e:
        st.error(f"Error extracting answer: {e}")
        return ""

def test_database_connection():
    """Test database connection"""
    try:
        with SessionLocal() as db:
            result = db.execute(text("SELECT 1")).fetchone()
            if result:
                return True, "Database connection successful"
            else:
                return False, "Database query returned no result"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

def get_relevant_pyqs_with_filters(session, query, filter_params, k=5):
    """Enhanced version of get_relevant_pyqs that applies filters before semantic search"""
    # Build filtered query
    db_query = session.query(PYQ)
    
    # Apply all filters
    for field, value in filter_params.items():
        if hasattr(PYQ, field):
            db_query = db_query.filter(getattr(PYQ, field) == value)
    
    # Get filtered PYQs
    filtered_pyqs = db_query.all()
    
    if not filtered_pyqs:
        return []
    
    # Create documents from filtered PYQs for semantic search
    docs = [
        Document(
            page_content=pyq.question,
            metadata={
                "year": pyq.year,
                "subject": pyq.subject,
                "sub_topic": pyq.sub_topic,
                "marks": pyq.marks,
                "college": getattr(pyq, 'college', None),
                "branch": getattr(pyq, 'branch', None),
                "unit": getattr(pyq, 'unit', None),
                "semester": getattr(pyq, 'semester', None),
            }
        )
        for pyq in filtered_pyqs
    ]
    
    # Perform semantic search on filtered documents
    try:
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embedding)
        results = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=20)
        return results
    except Exception as e:
        print(f"Semantic search error: {e}")
        # Fallback to simple text matching
        return docs[:k]

# Streamlit configuration
st.set_page_config(page_title="IntelliJect", layout="wide")
is_authenticated, user_data = init_auth()
# Enhanced CSS styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-card {
        background-color: #283250;
        color: #d6f5e3 !important;
        border-radius: 6px;
        padding: 10px 12px;
        margin: 10px 0 6px 0;
        font-size: 15px;
    }
    .highlight-answer {
        display: inline-block;
        background: #ffe44d;
        color: #232323 !important;
        border-radius: 5px;
        padding: 2px 6px;
        margin: 6px 0 2px 0;
        font-weight: 600;
        font-family: 'Georgia', serif;
        font-size: 15px;
    }
    .database-subtopic {
        background-color: #e8f5e8;
        color: #2d5a2d !important;
        border-radius: 4px;
        padding: 4px 8px;
        font-weight: bold;
    }
    .error-box {
        background-color: #ffebee;
        color: #d32f2f;
        padding: 10px;
        border-radius: 4px;
        border-left: 4px solid #d32f2f;
    }
    .section-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .question-preview {
        background-color: #1e293b;
        color: #f8fafc;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .smart-db-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_college' not in st.session_state:
    st.session_state.selected_college = "All Colleges"
if 'selected_branch' not in st.session_state:
    st.session_state.selected_branch = "All Branches"
if 'sdb_hierarchy_created' not in st.session_state:
    st.session_state.sdb_hierarchy_created = False
if 'sdb_context' not in st.session_state:
    st.session_state.sdb_context = {}
if 'study_plan' not in st.session_state:
    st.session_state.study_plan = None
if 'study_progress_days' not in st.session_state:
    st.session_state.study_progress_days = 0
if 'advanced_study_plan' not in st.session_state:
    st.session_state.advanced_study_plan = None
if 'advanced_progress_days' not in st.session_state:
    st.session_state.advanced_progress_days = 0
# ADD THESE NEW INITIALIZATIONS FOR SMART DATABASE BUILDER:
if 'sdb_selected_college' not in st.session_state:
    st.session_state.sdb_selected_college = "All Colleges"
if 'sdb_selected_branch' not in st.session_state:
    st.session_state.sdb_selected_branch = "All Branches"
if 'sdb_selected_course' not in st.session_state:
    st.session_state.sdb_selected_course = "All Courses"
# Main title

# Enhanced sidebar with detailed diagnostics
with st.sidebar:
    st.header("🔧 System Status")
    
    # Database connection test
    db_status, db_message = test_database_connection()
    if db_status:
        st.success(f"✅ {db_message}")
    else:
        st.error(f"❌ {db_message}")
        st.markdown(f"<div class='error-box'>Database connection failed. Please check your database configuration.</div>", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Sticky tab bar */
    .stTabs [role="tablist"] {
        
        position: sticky;
        top: 0;
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        left: 0;
        right: 0;
        background-color: black;
        z-index: 100;
        padding: 7px 12px;  /* controls vertical + horizontal padding */
        border-radius: 0;    /* remove rounded corners */
        display: flex;
        justify-content: center;  /* center the tabs */
        gap: 90px; /* space between tabs */
        box-shadow: 0 2px 6px rgba(0,0,0,0.4); /* subtle shadow */
    }

    /* Tab text color */
    .stTabs [role="tab"] {
        color: white !important;
        font-size:70px;!important
        
    }

    /* Active tab styling */
    .stTabs [aria-selected="true"] {
        
        color: white !important;
    
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Main application tabs
tab0,tab1, tab2, tab3, tab4 = st.tabs(["IntelliJect🎯","📚 Notes Matcher", "🏗️ Smart Database Builder", "🔍Search Engine", "📅 Study Planner"])


with tab0:
    video_file = open(r"C:\Users\imtey\OneDrive\aqsa\OneDrive\Desktop\background.mp4", "rb").read()
    video_bytes = base64.b64encode(video_file).decode()
    st.markdown(f"""
        <style>
        .video-bg {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100vw;
            min-height: 100vh;
            z-index: -1;
            object-fit: cover;
            opacity: 0.25; /* Adjust for readability */
        }}
        .stApp {{
            background: transparent !important;
        }}
        </style>

        <video autoplay loop muted playsinline class="video-bg">
            <source src="data:video/mp4;base64,{video_bytes}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .big-tagline {
            font-size: 4.0rem;
            font-weight: 600;
            margin: 100px 0 20px 0;
            color: #ffffff;
            letter-spacing: 2px;
            line-height: 1.1;
            font-family: 'Inter', sans-serif;
            text-align: left;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .sub-tagline {
            font-size: 2.5rem;    /* ~20px */
            font-weight: 500;      /* normal */
            color: #c2bfc2;        /* Tailwind gray-400 */
            line-height: 1.4;      /* good readability */
            margin-top: 0.5rem;    /* spacing from heading */
            margin-bottom: 0.5rem;
        }
        .blue-btn {
            background-color: #0099ff; /* bright blue */
            position: fixed;
            top:100px;
            left:2px;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 17px 34px;
            font-size: 25px;
            font-weight: 800;
            cursor: pointer;
            transition: background 0.3s ease;
            text-decoration: none; /* remove underline */
            
           
        }
        .blue-btn:hover {
            background-color: #007acc; /* darker blue on hover */
        }
        .features-container {
            display: flex;
            justify-content: space-around;
            text-align: center;
            padding: 50px 0;
            flex-wrap: wrap;
            color: black;
        }
        .feature {
            max-width: 250px;
            margin: 15px;
        }
        .feature-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: #1397ef; /* Tailwind indigo-500 */
            display: flex;
            align-items: center;
            justify-content: center;
            margin: auto;
            font-size: 24px;
            color: white;
        }
        .feature-title {
            font-size: 18px;
            color: #ffffff;
            font-weight: 600;
            margin-top: 15px;
            font-family: 'Inter', sans-serif;
        }
        .feature-desc {
            font-size: 14px;
            color: #c2bfc2; /* Tailwind gray-400 */
            margin-top: 5px;
            font-family: 'Inter', sans-serif;
        }
        .guide {
            font-size: 17px;
            color: #c2bfc2; 
            margin-top: 5px;
            font-family: 'Inter', sans-serif;
            text-align: center;
        }
        .guide-name {
            text-align: center;   /* ✅ Add this */
            margin-top: 90px; 
            margin-bottom: 30px;  /* optional: spacing below */
            font-size: 3.5rem;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 2px;
            line-height: 1.1;
            font-family: 'Inter', sans-serif;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .main-title {
            text-align: center;
            color: white;!important!;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 40px;
            font-family: 'Inter', sans-serif;
        }
        

        
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-tagline">Stop flipping docs, Start connecting dots.</div>', unsafe_allow_html=True)
    
    st.markdown(
    '<div class="sub-tagline">Get notes enriched with questions <br> AI didn’t guess, but contextually aligned for YOU.</div>',
    unsafe_allow_html=True
)
    #st.markdown('<a href="#" class="blue-btn">Get started for free</a>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0099ff; /* bright blue */
        color: white !important;   /* force white text */
        border: none;
        border-radius: 8px;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #007acc; /* darker blue on hover */
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

    # Streamlit button (now styled properly)
    st.button("Get started for free")
    st.markdown(
        """
        <style>
        .features-name {
            text-align: center;   /* ✅ Add this */
            margin-top: 130px; 
            margin-bottom: 30px;  /* optional: spacing below */
            font-size: 3.5rem;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 2px;
            line-height: 1.1;
            font-family: 'Inter', sans-serif;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="features-name">Exciting Features</div>', unsafe_allow_html=True)
    st.markdown(
    """
    
    <div class="features-container">
        <div class="feature">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Notes Matcher</div>
            <div class="feature-desc">Upload your notes—IntelliJect auto-links them with the right PYQs instantly.</div>
        </div>
        <div class="feature">
            <div class="feature-icon">🏗️</div>
            <div class="feature-title">Smart Database Builder</div>
            <div class="feature-desc">From raw question papers to organized databases in one click.</div>
        </div>
        <div class="feature">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Search Engine</div>
            <div class="feature-desc">Spot exam patterns, track frequently asked topics, and predict what really matters.</div>
        </div>
        <div class="feature">
            <div class="feature-icon">📅</div>
            <div class="feature-title">Study Planner</div>
            <div class="feature-desc">Turn chaos into a clear roadmap with our smart study planner.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown('<div class="guide-name">It’s your time to shine!</div>', unsafe_allow_html=True)
    st.markdown(
    '<div class="guide">Start by navigating to the "📚 Notes Matcher" tab to upload your notes and find relevant questions. For educators and administrators, explore the "🏗️ Smart Database Builder" tab to contribute to our growing question bank. Finally, check out the "📊 Smart Search Engine " tab for quick insights into question trends and statistics.<br> Let’s make studying smarter with IntelliJect! 🚀</div>',
    unsafe_allow_html=True
)
    st.markdown("---")

st.markdown("""
    <style>
        .footer {
            background-color: #000;  /* Black footer */
            padding: 3rem 1rem;
            color: #fff;
            font-family: 'Inter', sans-serif;
        }
        .footer-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 2rem;
            max-width: 1200px;
            margin: auto;
        }
        .footer-section h3 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #fff;
        }
        .footer-section p, 
        .footer-section a {
            color: #bbb;
            text-decoration: none;
            font-size: 0.95rem;
        }
        .footer-section a:hover {
            color: #fff;
        }
        .social-icons {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }
        .social-icons svg {
            width: 24px;
            height: 24px;
            fill: #bbb;
            transition: fill 0.3s;
        }
        .social-icons a:hover svg {
            fill: #fff;
        }
        .footer-bottom {
            border-top: 1px solid #333;
            margin-top: 2rem;
            padding-top: 1rem;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            font-size: 0.9rem;
            color: #aaa;
        }
        .footer-bottom a {
            margin-left: 1rem;
            color: #bbb;
            text-decoration: none;
        }
        .footer-bottom a:hover {
            color: #fff;
        }
    </style>

    <div class="footer">
        <div class="footer-container">
            <!-- Left: Logo + tagline + icons -->
            <div class="footer-section">
                <h2><b>Intelliject</b></h2>
                <p>Boost your productivity with AI-enhanced notes and insights.</p>
                <div class="social-icons">
                    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24"><path d="M23.954 4.569c-.885.392-1.83.656-2.825.775 1.014-.611 1.794-1.574 2.163-2.724-.951.555-2.005.959-3.127 1.184-.896-.959-2.178-1.555-3.594-1.555-2.717 0-4.924 2.206-4.924 4.924 0 .39.045.765.127 1.124-4.09-.205-7.719-2.165-10.148-5.144-.424.729-.666 1.577-.666 2.475 0 1.708.87 3.216 2.188 4.099-.807-.026-1.566-.248-2.229-.616v.062c0 2.385 1.693 4.374 3.946 4.827-.413.111-.849.171-1.296.171-.317 0-.626-.031-.928-.088.627 1.956 2.444 3.377 4.6 3.419-1.68 1.318-3.809 2.106-6.102 2.106-.396 0-.787-.023-1.174-.068 2.179 1.398 4.768 2.21 7.557 2.21 9.054 0 14.001-7.496 14.001-13.986 0-.213-.004-.425-.014-.637.962-.695 1.8-1.562 2.46-2.549z"/></svg></a>
                    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24"><path d="M19.615 3.184C21.403 3.672 22.67 5.07 23.156 6.857 23.999 9.994 24 12 24 12s0 2.006-.844 5.143c-.486 1.787-1.753 3.185-3.541 3.673C17.478 21.66 12 21.66 12 21.66s-5.478 0-8.615-.844c-1.788-.488-3.055-1.886-3.541-3.673C-.001 14.006 0 12 0 12s-.001-2.006.844-5.143C1.33 5.07 2.597 3.672 4.385 3.184 7.522 2.34 13 2.34 13 2.34s5.478 0 8.615.844zm-11.49 5.495v6.642l6.479-3.321-6.479-3.321z"/></svg></a>
                    <a href="#"><svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.372 0 0 5.373 0 12c0 5.303 3.438 9.8 8.207 11.387.6.111.793-.261.793-.579 0-.285-.01-1.041-.016-2.042-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.334-1.756-1.334-1.756-1.09-.744.082-.729.082-.729 1.205.085 1.838 1.237 1.838 1.237 1.07 1.834 2.809 1.305 3.494.997.108-.774.418-1.305.762-1.605-2.665-.305-5.467-1.334-5.467-5.931 0-1.31.469-2.381 1.236-3.221-.124-.303-.536-1.527.116-3.181 0 0 1.008-.323 3.3 1.23a11.5 11.5 0 0 1 3.003-.404c1.019.005 2.046.138 3.003.404 2.292-1.553 3.3-1.23 3.3-1.23.652 1.654.24 2.878.117 3.181.768.84 1.236 1.911 1.236 3.221 0 4.609-2.807 5.624-5.479 5.921.429.369.823 1.102.823 2.222 0 1.605-.015 2.896-.015 3.286 0 .32.192.694.799.576C20.565 21.796 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg></a>
                </div>
                <br>
                <p>© 2025 IntelliJect</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


    
# TAB 1: Notes Matcher
with tab1:
    st.markdown("### 📑 Upload Notes and Match PYQs")
    
    # Enhanced filter section with cascading dropdowns
    st.markdown("#### 🎯 Filter Configuration")
    
    # Row 1: College and Course
    col1, col2 = st.columns(2)
    with col1:
        college_options = ["All Colleges"] + list(UNIVERSITY_CONFIG.keys()) + ["Other"]
        college = st.selectbox(
            "🏫 Select College", 
            college_options,
            key="filter_college",
            index=college_options.index(st.session_state.selected_college) if st.session_state.selected_college in college_options else 0
        )
        
        # Update session state when college changes
        if college != st.session_state.selected_college:
            st.session_state.selected_college = college
            st.session_state.selected_branch = "All Branches"
            st.rerun()
    
    with col2:
        course = st.selectbox(
            "🎓 Select Course", 
            ["All Courses", "B.Tech", "BE", "B.Com", "BDS", "M.Tech", "Other"],
            key="filter_course"
        )
    
    # Row 2: Branch and Semester Type
    col3, col4 = st.columns(2)
    with col3:
        available_branches = get_available_branches(college)
        branch = st.selectbox(
            "🌿 Select Branch", 
            available_branches,
            key="filter_branch",
            index=available_branches.index(st.session_state.selected_branch) if st.session_state.selected_branch in available_branches else 0
        )
        
        if branch != st.session_state.selected_branch:
            st.session_state.selected_branch = branch
    
    with col4:
        semester_type = st.selectbox(
            "📝 Exam Type", 
            ["All Types", "MID SEMESTER", "END SEMESTER", "QUIZ", "ASSIGNMENT"],
            key="filter_semester_type",
            help="Filter questions by exam type (affects database query)"
        )
    
    # Row 3: Academic Semester and Unit
    col_sem, col_unit = st.columns(2)
    with col_sem:
        available_semesters = get_available_semesters(college, branch)
        if available_semesters:
            academic_semester = st.selectbox(
                "📅 Academic Semester",
                ["All Semesters"] + available_semesters,
                key="filter_academic_semester",
                help="Select semester to filter subject list (does not affect question search)"
            )
        else:
            academic_semester = st.selectbox(
                "📅 Academic Semester",
                ["All Semesters"],
                key="filter_academic_semester",
                help="Select semester to filter subject list (does not affect question search)"
            )
    
    with col_unit:
        unit = st.selectbox(
            "📖 Unit (Optional)", 
            ["All Units", "Unit 1", "Unit 2", "Unit 3", "Unit 4"],
            key="filter_unit"
        )
    
    # Row 4: Subject
    col_subject, col_info = st.columns([2, 1])
    with col_subject:
        semester_filter = academic_semester if academic_semester != "All Semesters" else None
        available_subjects = get_available_subjects(college, branch, semester_filter)
        
        if available_subjects:
            subject = st.selectbox(
                "📚 Select Subject", 
                available_subjects,
                key="filter_subject"
            )
        else:
            subject = st.selectbox(
                "📚 Select Subject", 
                ["Cyber Security", "Environmental Sciences", "Probability and Statistics", 
                 "Applied Mathematics","Introduction to Data Science", "Data Structures", "Computer Networks", "Other"],
                key="filter_subject"
            )
            if college != "All Colleges" and branch != "All Branches":
                st.warning(f"⚠️ No predefined subjects for {college} - {branch} - {academic_semester}. Showing default options.")
    
    with col_info:
        if available_subjects:
            st.metric("📊 Available Subjects", len(available_subjects))
        else:
            st.metric("📊 Available Subjects", "Default")
    
    # Show current selection summary
    if college != "All Colleges" or branch != "All Branches":
        with st.expander("🔍 Current Selection Summary", expanded=False):
            st.markdown(f"""
            **University:** {college}  
            **Branch:** {branch}  
            **Academic Semester (for subjects):** {academic_semester if 'academic_semester' in locals() else 'All Semesters'}  
            **Available Subjects:** {len(available_subjects) if available_subjects else 0}  
            **Selected Subject:** {subject}  
            **Exam Type (for filtering):** {semester_type}
            """)
    
    # File upload section
    st.markdown("#### 📁 File Upload")
    uploaded_file = st.file_uploader("📑 Upload your notes PDF", type=["pdf"])

    # Only show button if database is connected
    if not db_status:
        st.error("❌ Cannot proceed - database connection failed. Please check your database setup.")
        st.stop()

    if uploaded_file and subject:
        # Show current filter configuration
        with st.expander("🔍 Database Filter Settings", expanded=False):
            filter_info = f"""
            **College:** {college}  
            **Course:** {course}  
            **Branch:** {branch}  
            **Exam Type:** {semester_type}  
            **Subject:** {subject}  
            **Unit:** {unit}
            
            Note: Academic Semester ({academic_semester if 'academic_semester' in locals() else 'All'}) was used only for subject selection and does not affect question search.
            """
            st.markdown(filter_info)
        
        match_button = st.button("🔍 Match PYQs with Filters", type="primary", use_container_width=True)
        
        if not match_button:
            st.info("👆 Click 'Match PYQs with Filters' to process your PDF and find relevant questions based on your filter criteria.")
        else:
            # Create filter dictionary for database queries
            filter_params = {}
            if college != "All Colleges":
                filter_params['college'] = college
            if course != "All Courses":
                filter_params['course'] = course
            if branch != "All Branches":
                filter_params['branch'] = branch
            if semester_type != "All Types":
                filter_params['semester'] = semester_type
            if unit != "All Units":
                filter_params['unit'] = unit.replace("Unit ", "")
            
            filter_params['subject'] = subject
            
            # Display active filters
            if filter_params:
                st.info(f"🎯 Active database filters: {', '.join([f'{k}: {v}' for k, v in filter_params.items()])}")
                st.info(f"📚 Academic semester '{academic_semester if 'academic_semester' in locals() else 'All'}' was used for subject selection only")
            
            # Process PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pdf_path = tmp_file.name

            st.success(f"✅ PDF '{uploaded_file.name}' loaded for processing with filters applied.")

            st.subheader("📑 Extracting and Chunking Notes...")
            
            try:
                text_chunks = extract_text_from_pdf(tmp_pdf_path)
            except Exception as e:
                st.error(f"❌ Could not extract content from PDF: {e}")
                st.stop()

            if not text_chunks:
                st.error("❌ Could not extract content from the PDF.")
                st.stop()
            else:
                st.success(f"✅ Successfully extracted {len(text_chunks)} chunks.")

                # Test subject data availability with filters
                try:
                    with SessionLocal() as db:
                        query = db.query(PYQ)
                        
                        for field, value in filter_params.items():
                            if hasattr(PYQ, field):
                                query = query.filter(getattr(PYQ, field) == value)
                        
                        subject_count = query.count()
                        
                        if subject_count == 0:
                            st.error(f"❌ No PYQs found matching your filter criteria.")
                            
                            # Show available data for debugging
                            st.markdown("#### 📊 Available Data in Database:")
                            
                            available_subjects = db.execute(text("SELECT DISTINCT subject FROM pyqs")).fetchall()
                            if available_subjects:
                                subjects_list = [subj[0] for subj in available_subjects]
                                st.info(f"**Subjects:** {', '.join(subjects_list)}")
                            
                            if hasattr(PYQ, 'college'):
                                colleges = db.execute(text("SELECT DISTINCT college FROM pyqs WHERE college IS NOT NULL")).fetchall()
                                if colleges:
                                    college_list = [col[0] for col in colleges]
                                    st.info(f"**Colleges:** {', '.join(college_list)}")
                            
                            if hasattr(PYQ, 'branch'):
                                branches = db.execute(text("SELECT DISTINCT branch FROM pyqs WHERE branch IS NOT NULL")).fetchall()
                                if branches:
                                    branch_list = [br[0] for br in branches]
                                    st.info(f"**Branches:** {', '.join(branch_list)}")
                            
                            st.stop()
                        else:
                            st.success(f"📚 Found {subject_count} PYQs matching your filter criteria")
                            
                except Exception as e:
                    st.error(f"❌ Could not check filtered data: {e}")
                    st.stop()

                try:
                    pdf_doc = fitz.open(tmp_pdf_path)
                    num_pages = pdf_doc.page_count
                except Exception as e:
                    st.error(f"❌ Could not open PDF with PyMuPDF: {e}")
                    st.stop()

                if len(text_chunks) != num_pages:
                    st.warning(f"⚠️ Number of text chunks ({len(text_chunks)}) does not match number of PDF pages ({num_pages}). Highlighting may be inaccurate.")

                # Process chunks with enhanced filtering
                for i, chunk in enumerate(text_chunks):
                    col_img, col_pyqs = st.columns([1.5, 1])
                    answers_to_highlight = []

                    with col_pyqs:
                        st.markdown(f"### 📄 Page {i+1}")
                        
                        try:
                            with SessionLocal() as session:
                                related_qs = get_relevant_pyqs_with_filters(session, chunk, filter_params)
                                
                                if related_qs:
                                    subtopic = related_qs[0].metadata.get('sub_topic', 'General')
                                    st.markdown(
                                        f"<span style='font-size:18px;font-weight:bold;'>🔎 Subtopic: "
                                        f"<span class='database-subtopic'>{subtopic}</span></span>", 
                                        unsafe_allow_html=True
                                    )
                                    
                                    st.markdown(f"<small>🎯 Matches: {len(related_qs)} questions with applied filters</small>", unsafe_allow_html=True)
                                else:
                                    subtopic = "No matches found"
                                    st.markdown(
                                        f"<span style='font-size:18px;font-weight:bold;'>🔎 Subtopic: {subtopic}</span>", 
                                        unsafe_allow_html=True
                                    )
                                    
                        except Exception as e:
                            st.error(f"❌ Database query failed: {e}")
                            related_qs = []

                        if related_qs:
                            for idx, q in enumerate(related_qs[:3]):
                                answer_text = extract_answer_from_chunk(chunk, q.page_content)
                                if answer_text:
                                    for sent in sent_tokenize(answer_text):
                                        sent_clean = sent.strip()
                                        if sent_clean:
                                            answers_to_highlight.append(sent_clean)

                                metadata_info = []
                                for key in ['sub_topic', 'marks', 'year', 'college', 'branch', 'unit', 'semester']:
                                    value = q.metadata.get(key, 'N/A')
                                    if value and value != 'N/A':
                                        metadata_info.append(f"{key.replace('_', ' ').title()}: {value}")
                                
                                metadata_str = " | ".join(metadata_info) if metadata_info else "No additional metadata"

                                st.markdown(
                                    f"<div class='question-card'>"
                                    f"❓ <b>Q{idx+1}:</b> {q.page_content}<br>"
                                    f"<span style='font-size:12px;opacity:0.8;'>{metadata_str}</span><br>"
                                    f"<span class='highlight-answer'><b>📌 Answer:</b> {answer_text if answer_text else '(No direct answer found)'}</span>"
                                    f"</div>", unsafe_allow_html=True
                                )
                                st.markdown("---", unsafe_allow_html=True)
                        else:
                            st.info("❗ No relevant PYQs found for this chunk with current filters.")

                    with col_img:
                        try:
                            if i >= num_pages:
                                st.warning(f"Chunk index {i} out of PDF pages range ({num_pages}). Skipping highlighting.")
                                continue
                                
                            page = pdf_doc[i]

                            highlight_count = 0
                            
                            for answer_frag in answers_to_highlight:
                                if answer_frag and len(answer_frag.strip()) > 3:
                                    try:
                                        rects = page.search_for(answer_frag)
                                        if rects:
                                            for rect in rects:
                                                annot = page.add_highlight_annot(rect)
                                                annot.set_colors(stroke=(1, 1, 0))
                                                annot.update()
                                                highlight_count += 1
                                    except Exception as highlight_error:
                                        pass

                            pix = page.get_pixmap(dpi=150)
                            img_data = pix.pil_tobytes(format="PNG")
                            img = Image.open(BytesIO(img_data))
                            
                            st.image(img, caption=f"PDF Page {i+1} ({highlight_count} highlights)", use_container_width=True)
                            
                            if highlight_count > 0:
                                st.success(f"✨ {highlight_count} answer segments highlighted on this page")
                            else:
                                st.info("💡 No answer text found on this page to highlight")

                        except Exception as e:
                            st.warning(f"Could not render PDF page {i+1}: {e}")
                            st.text_area(f"Page {i+1} Text Content", chunk[:500] + "...", height=300)

                # Clean up
                try:
                    pdf_doc.close()
                    import os
                    os.unlink(tmp_pdf_path)
                except:
                    pass

# TAB 2: Smart Database Builder
with tab2:
    # Add this at the beginning of tab2
    if not is_authenticated:
        st.info("🔒 Please login to access the Smart Database Builder")
        st.stop()
    
    st.markdown("""
        <div class="smart-db-header">
            <h2>🏗️ Smart Database Builder</h2>
            <p>Intelligent Question Paper Processing & Database Management</p>
        </div>
    """, unsafe_allow_html=True)
    if 'enhanced_sdb' not in st.session_state:
            from enhanced_smart_database_builder import EnhancedSmartDatabaseBuilder
            st.session_state.enhanced_sdb = EnhancedSmartDatabaseBuilder()
        
    sdb = st.session_state.enhanced_sdb
    sdb_tab1, sdb_tab2, sdb_tab3= st.tabs(["Setup Hierarchy", "Preview Before Upload", "Processing Status"])
    # Sub-tabs for Smart Database Builder
    # sdb_tab1, sdb_tab2, sdb_tab3 = st.tabs(["📁 Setup Hierarchy", "📄 Upload Documents", "📊 Processing Status"])
    
    with sdb_tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("🏫 Setup Academic Hierarchy")
    # Initialize enhanced Smart Database Builder
        
         
        # Initialize session state for SDB if not exists
        if 'sdb_selected_college' not in st.session_state:
            st.session_state.sdb_selected_college = "All Colleges"
        if 'sdb_selected_branch' not in st.session_state:
            st.session_state.sdb_selected_branch = "All Branches"
        if 'sdb_selected_course' not in st.session_state:
            st.session_state.sdb_selected_course = "All Courses"
        
        # Row 1: College and Course
        col1, col2 = st.columns(2)
        
        with col1:
            college_options = ["All Colleges"] + list(UNIVERSITY_CONFIG.keys()) + ["Other"]
            college = st.selectbox(
                "🏫 College Name", 
                college_options,
                key="sdb_college_select",
                index=college_options.index(st.session_state.sdb_selected_college) if st.session_state.sdb_selected_college in college_options else 0,
                help="Select your college/university"
            )
            
            # Update session state when college changes
            if college != st.session_state.sdb_selected_college:
                st.session_state.sdb_selected_college = college
                st.session_state.sdb_selected_branch = "All Branches"
                st.rerun()
        
        with col2:
            course_options = ["All Courses", "B.Tech", "BE", "B.Com", "BDS", "M.Tech", "MBA", "MCA", "Other"]
            course = st.selectbox(
                "🎓 Course", 
                course_options,
                key="sdb_course_select",
                index=course_options.index(st.session_state.sdb_selected_course) if st.session_state.sdb_selected_course in course_options else 0,
                help="Select your course/degree"
            )
            
            if course != st.session_state.sdb_selected_course:
                st.session_state.sdb_selected_course = course
        
        # Row 2: Branch and Academic Year
        col3, col4 = st.columns(2)
        
        with col3:
            available_branches = get_available_branches(college)
            branch = st.selectbox(
                "🌿 Branch", 
                available_branches,
                key="sdb_branch_select",
                index=available_branches.index(st.session_state.sdb_selected_branch) if st.session_state.sdb_selected_branch in available_branches else 0,
                help="Select your branch/specialization"
            )
            
            if branch != st.session_state.sdb_selected_branch:
                st.session_state.sdb_selected_branch = branch
        
        with col4:
            from datetime import datetime
            current_year = datetime.now().year
            year_options = [f"{current_year-i}-{current_year-i+1}" for i in range(5)]
            year = st.selectbox(
                "📅 Academic Year", 
                year_options,
                key="sdb_year_select",
                index=0,
                help="Select the academic year"
            )
        
        # Row 3: Academic Semester and Semester Type
        col5, col6 = st.columns(2)
        
        with col5:
            available_semesters = get_available_semesters(college, branch)
            if available_semesters:
                academic_semester = st.selectbox(
                    "📚 Academic Semester",
                    ["All Semesters"] + available_semesters,
                    key="sdb_academic_semester",
                    help="Select semester to filter subject list"
                )
            else:
                academic_semester = st.selectbox(
                    "📚 Academic Semester",
                    ["All Semesters", "Semester 1", "Semester 2", "Semester 3", "Semester 4", 
                     "Semester 5", "Semester 6", "Semester 7", "Semester 8"],
                    key="sdb_academic_semester",
                    help="Select semester to filter subject list"
                )
        
        with col6:
            semester_type = st.selectbox(
                "📝 Semester Type", 
                ["mid-sem", "end-sem", "quiz", "assignment", "CAT", "FAT"],
                key="sdb_semester_type",
                help="Type of examination/assessment"
            )
        
        # Row 4: Subject Selection
        col7, col8 = st.columns([2, 1])
        
        with col7:
            semester_filter = academic_semester if academic_semester != "All Semesters" else None
            available_subjects = get_available_subjects(college, branch, semester_filter)
            
            if available_subjects:
                subject = st.selectbox(
                    "📚 Subject", 
                    available_subjects,
                    key="sdb_subject_select",
                    help="Select the subject for question paper processing"
                )
            else:
                # Default subjects if no specific data available
                default_subjects = [
                    "Applied Mathematics", "Cyber Security", "Environmental Sciences", 
                    "Probability and Statistics", "Data Structures", "Computer Networks",
                    "Database Management Systems", "Operating Systems", "Machine Learning",
                    "Artificial Intelligence", "Software Engineering", "Computer Graphics",
                    "Digital Signal Processing", "Microprocessors","Introduction to Data Science", "Other"
                ]
                subject = st.selectbox(
                    "📚 Subject", 
                    default_subjects,
                    key="sdb_subject_select",
                    help="Select the subject for question paper processing"
                )
                
                if college != "All Colleges" and branch != "All Branches":
                    st.warning(f"⚠️ No predefined subjects for {college} - {branch} - {academic_semester}. Showing default options.")
        
        with col8:
            if available_subjects:
                st.metric("📊 Available Subjects", len(available_subjects))
            else:
                st.metric("📊 Available Subjects", "Default List")
        
        # Show current selection summary
        if college != "All Colleges" or branch != "All Branches":
            with st.expander("🔍 Current Hierarchy Summary", expanded=False):
                st.markdown(f"""
                **College:** {college}  
                **Course:** {course}  
                **Branch:** {branch}  
                **Academic Year:** {year}  
                **Academic Semester:** {academic_semester}  
                **Semester Type:** {semester_type}  
                **Subject:** {subject}  
                **Available Subjects:** {len(available_subjects) if available_subjects else len(default_subjects) if 'default_subjects' in locals() else 0}
                """)
        
        # Additional metadata fields (optional)
        st.markdown("#### 📝 Additional Information (Optional)")
        col9, col10 = st.columns(2)
        
        with col9:
            unit_filter = st.selectbox(
                "📖 Unit Focus (Optional)", 
                ["All Units", "Unit 1", "Unit 2", "Unit 3", "Unit 4", "Unit 5"],
                key="sdb_unit_filter",
                help="Specify if questions belong to a particular unit"
            )
        
        with col10:
            question_type = st.selectbox(
                "❓ Question Type",
                ["All Types", "MCQ", "Short Answer", "Long Answer", "Numerical", "Coding", "Theory"],
                key="sdb_question_type",
                help="Primary type of questions in the paper"
            )
        
        # Validation and confirmation
        st.markdown("---")
        
        # Check if all required fields are filled
        required_fields = [college, course, branch, year, academic_semester, semester_type, subject]
        all_filled = all(field and field not in ["All Colleges", "All Courses", "All Branches", "All Semesters"] 
                        for field in required_fields[:3]) and all(required_fields[3:])
        
        if not all_filled:
            st.warning("⚠️ Please select specific values for College, Course, Branch, and all other required fields.")
        
        col_confirm, col_reset = st.columns([2, 1])
        
        with col_confirm:
            confirm_button = st.button(
                "✅ Confirm Hierarchy", 
                type="primary", 
                key="sdb_confirm_hierarchy",
                disabled=not all_filled,
                use_container_width=True
            )
            
            if confirm_button:
                # Store the configuration
                st.session_state.sdb_context = {
                    'college': college,
                    'course': course,
                    'subject': subject,
                    'branch': branch,
                    'year': year,
                    'academic_semester': academic_semester,
                    'semester': semester_type,
                    'unit': unit_filter.replace("Unit ", "") if unit_filter != "All Units" else None,
                    'question_type': question_type if question_type != "All Types" else None
                }
                st.session_state.sdb_hierarchy_created = True
                
                # Success message with details
                st.success("✅ Academic hierarchy configured successfully!")
                
                # Show configured details
                st.markdown("""
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 15px; margin: 10px 0;">
                <h4 style="color: #155724; margin: 0 0 10px 0;">📋 Configuration Saved:</h4>
                """, unsafe_allow_html=True)
                
                config_display = []
                for key, value in st.session_state.sdb_context.items():
                    if value:
                        display_key = key.replace('_', ' ').title()
                        config_display.append(f"<strong>{display_key}:</strong> {value}")
                
                st.markdown("<br>".join(config_display), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.info("👉 Now proceed to the 'Upload Documents' tab to upload question papers.")
        
        with col_reset:
            if st.button("🔄 Reset", key="sdb_reset_hierarchy", use_container_width=True):
                # Reset all session state variables
                for key in ['sdb_context', 'sdb_hierarchy_created', 'sdb_selected_college', 
                           'sdb_selected_branch', 'sdb_selected_course']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with sdb_tab2:
        if not st.session_state.get('sdb_hierarchy_created', False):
            st.warning("⚠️ Please setup the academic hierarchy first in the Setup tab")
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.header("📄 Upload Academic Documents")
            
            # Display current context
            with st.expander("📋 Current Configuration", expanded=False):
                context = st.session_state.get('sdb_context', {})
                for key, value in context.items():
                    if value:
                        display_key = key.replace('_', ' ').title()
                        st.write(f"**{display_key}:** {value}")
            
            # File upload section
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Syllabus Upload (Optional)")
                syllabus_file = st.file_uploader(
                    "Upload Syllabus (PDF/Image)", 
                    type=['pdf', 'png', 'jpg', 'jpeg'],
                    key="sdb_syllabus",
                    help="Upload syllabus to help with question categorization"
                )
                
                if syllabus_file:
                    st.success(f"✅ Syllabus file uploaded: {syllabus_file.name}")
            
            with col2:
                st.subheader("❓ Question Papers Upload")
                question_files = st.file_uploader(
                    "Upload Question Papers (PDF/Images)", 
                    type=['pdf', 'png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    key="sdb_questions",
                    help="Upload multiple question paper files"
                )
                
                if question_files:
                    st.success(f"✅ {len(question_files)} question file(s) uploaded")
                    
                    # Show uploaded files
                    with st.expander("📁 Uploaded Files", expanded=False):
                        for i, file in enumerate(question_files, 1):
                            file_size = len(file.getvalue()) / 1024  # Size in KB
                            st.write(f"{i}. **{file.name}** ({file_size:.1f} KB)")
            
            # Process button
            if question_files and st.button("🚀 Process Question Papers", type="primary", key="process_questions"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_questions = []
                
                for i, file in enumerate(question_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i + 1) / len(question_files))
                    
                    # Extract text based on file type
                    if file.type == "application/pdf":
                        text = smart_db.extract_text_from_pdf_enhanced(file)
                    else:
                        text = smart_db.extract_text_from_image(file)
                    
                    if text.strip():
                        # Parse questions using LLM
                        questions = smart_db.parse_questions_with_llm(text, st.session_state.sdb_context)
                        
                        # Validate questions
                        valid_questions = []
                        for q in questions:
                            is_valid, reason = smart_db.validate_question_data(q)
                            if is_valid:
                                valid_questions.append(q)
                            # else:
                            #     st.warning(f"Skipped invalid question from {file.name}: {reason}")
                        
                        all_questions.extend(valid_questions)
                    else:
                        st.warning(f"No text extracted from {file.name}")
                
                # Store questions in session state for review
                st.session_state.extracted_questions = all_questions
                
                # Display extracted questions for review
                if all_questions:
                    st.subheader("📝 Extracted Questions Preview")
                    
                    # Summary metrics
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("📊 Total Questions", len(all_questions))
                    # with col_metrics2:
                    #     total_marks = sum(q.get('marks', 1) for q in all_questions)
                    #     st.metric("🎯 Total Marks", f"{total_marks}")
                    with col_metrics3:
                        unique_topics = len(set(q.get('sub_topic', 'Unknown') for q in all_questions))
                        st.metric("🏷️ Unique Topics", unique_topics)
                    
                    # Show sample questions
                    for i, q in enumerate(all_questions[:5]):
                        with st.expander(f"Question {i+1}: {q['question'][:100]}..."):
                            st.markdown(f"""
                            <div class="question-preview">
                            <strong>Question:</strong> {q['question']}<br>
                            <strong>Sub-topic:</strong> {q.get('sub_topic', 'Not identified')}<br>
                            <strong>Marks:</strong> {q.get('marks', 1)}<br>
                            <strong>Unit:</strong> {q.get('unit', 'Not identified')}<br>
                            <strong>Type:</strong> {q.get('question_type', 'Not specified')}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if len(all_questions) > 5:
                        st.info(f"... and {len(all_questions) - 5} more questions")
                    
                    # Confirmation buttons
                    col1, col2, col3 = st.columns(3)
                    
                    # with col1:
                    #     if st.button("✅ Save All to Database", type="primary", key="save_to_db"):
                    #         with st.spinner("Saving questions to database..."):
                    #             added, updated, skipped = smart_db.save_questions_to_database(
                    #                 all_questions, st.session_state.sdb_context
                    #             )
                    def enhanced_sdb_tab2_save_logic():
                        """Enhanced save logic with hash generation for sdb_tab2"""
                        
                        # In your sdb_tab2, replace the save button section with this:
                        with col1:
                            if st.button("✅ Save All to Database", type="primary", key="save_to_db"):
                                with st.spinner("Processing questions with hash generation..."):
                                    # Generate hashes and check for duplicates
                                    processed_questions = []
                                    duplicate_count = 0
                                    
                                    progress_bar = st.progress(0)
                                    for i, q in enumerate(all_questions):
                                        progress_bar.progress((i + 1) / len(all_questions))
                                        
                                        # Generate hash for each question
                                        question_hash = generate_question_hash(q.get('question', ''))
                                        
                                        if not question_hash:
                                            continue
                                        
                                        # Check if hash already exists in current batch
                                        existing_hashes = [pq.get('question_hash') for pq in processed_questions]
                                        if question_hash in existing_hashes:
                                            duplicate_count += 1
                                            continue
                                        
                                        # Check if hash exists in database
                                        if check_question_exists_by_hash(question_hash, st.session_state.sdb_context.get('subject')):
                                            duplicate_count += 1
                                            continue
                                        
                                        # Add hash to question data
                                        q['question_hash'] = question_hash
                                        processed_questions.append(q)
                                    
                                    progress_bar.progress(1.0)
                                    
                                    # Save to database
                                    with st.spinner("Saving to database..."):
                                        added, updated, skipped = save_questions_to_database_with_hash(
                                            processed_questions, st.session_state.sdb_context
                                        )
                                
                                # Display results with hash information
                                st.markdown(f"""
                                <div class="success-box">
                                <h4>📊 Processing Complete!</h4>
                                <ul>
                                <li>🔍 <strong>{len(all_questions)}</strong> questions processed</li>
                                <li>🏷️ <strong>{len(processed_questions)}</strong> questions with valid hashes</li>
                                <li>🔄 <strong>{duplicate_count}</strong> duplicates detected and skipped</li>
                                <li>✅ <strong>{added}</strong> new questions added</li>
                                <li>🔄 <strong>{updated}</strong> existing questions updated (frequency increased)</li>
                                <li>⏭️ <strong>{skipped}</strong> questions skipped due to errors</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show some sample hashes for verification
                                if processed_questions:
                                    with st.expander("🔍 Sample Question Hashes (for debugging)", expanded=False):
                                        for i, q in enumerate(processed_questions[:3]):
                                            st.write(f"**Question {i+1}:**")
                                            st.write(f"Text: {q['question'][:100]}...")
                                            st.write(f"Hash: `{q['question_hash']}`")
                                            st.write("---")
                                
                                if 'extracted_questions' in st.session_state:
                                    del st.session_state.extracted_questions

                    # Additional utility function for manual hash checking
                    def show_question_hash_info(question_text: str):
                        """Utility function to show hash information for a question"""
                        if question_text:
                            question_hash = generate_question_hash(question_text)
                            st.write(f"**Question Hash:** `{question_hash}`")
                            
                            # Check if exists in database
                            exists = check_question_exists_by_hash(question_hash)
                            if exists:
                                st.warning("⚠️ This question hash already exists in database")
                            else:
                                st.success("✅ This is a new question hash")

                    # Function to add hash column to existing questions (migration helper)
                    def add_hashes_to_existing_questions():
                        """Add question_hash to existing questions in database"""
                        try:
                            with SessionLocal() as db:
                                # Get questions without hashes
                                questions = db.query(PYQ).filter(
                                    (PYQ.question_hash == None) | (PYQ.question_hash == '')
                                ).all()
                                
                                updated_count = 0
                                for question in questions:
                                    if question.question:
                                        question_hash = generate_question_hash(question.question)
                                        question.question_hash = question_hash
                                        updated_count += 1
                                
                                db.commit()
                                return updated_count
                                
                        except Exception as e:
                            print(f"Error adding hashes to existing questions: {e}")
                            return 0        
                            st.markdown(f"""
                            <div class="success-box">
                            <h4>📊 Processing Complete!</h4>
                            <ul>
                            <li>✅ <strong>{added}</strong> new questions added</li>
                            <li>🔄 <strong>{updated}</strong> existing questions updated (frequency increased)</li>
                            <li>⏭️ <strong>{skipped}</strong> questions skipped</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if 'extracted_questions' in st.session_state:
                                del st.session_state.extracted_questions
                    
                    
                
                else:
                    st.error("❌ No valid questions found in uploaded files")
            
            # Review mode for individual question editing
            if st.session_state.get('review_mode', False) and st.session_state.get('extracted_questions', []):
                st.markdown("---")
                st.subheader("📝 Review & Edit Questions")
                
                for i, q in enumerate(st.session_state.extracted_questions):
                    with st.expander(f"Edit Question {i+1}", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            edited_question = st.text_area(
                                "Question Text:",
                                value=q['question'],
                                key=f"edit_q_{i}",
                                height=100
                            )
                            edited_subtopic = st.text_input(
                                "Sub-topic:",
                                value=q.get('sub_topic', ''),
                                key=f"edit_st_{i}"
                            )
                        
                        with col2:
                            edited_marks = st.number_input(
                                "Marks:",
                                value=float(q.get('marks', 1)),
                                min_value=0.5,
                                max_value=100.0,
                                step=0.5,
                                key=f"edit_m_{i}"
                            )
                            edited_unit = st.text_input(
                                "Unit:",
                                value=q.get('unit', ''),
                                key=f"edit_u_{i}"
                            )
                        
                        # Update the question data
                        st.session_state.extracted_questions[i].update({
                            'question': edited_question,
                            'sub_topic': edited_subtopic,
                            'marks': edited_marks,
                            'unit': edited_unit
                        })
                
                # Save edited questions
                if st.button("💾 Save Edited Questions", type="primary", key="save_edited"):
                    with st.spinner("Saving edited questions..."):
                        added, updated, skipped = smart_db.save_questions_to_database(
                            st.session_state.extracted_questions, st.session_state.sdb_context
                        )
                    
                    st.success(f"✅ Saved: {added} new, {updated} updated, {skipped} skipped")
                    
                    st.session_state.review_mode = False
                    if 'extracted_questions' in st.session_state:
                        del st.session_state.extracted_questions
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with sdb_tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("📊 Smart Database Status & JSON Processor")
        
        # Add JSON processing section
        st.subheader("📄 Process Scanned Data to JSON")
        
        col_upload, col_process = st.columns([1, 1])
        
        with col_upload:
            st.markdown("#### 📁 Upload Scanned Documents")
            scanned_files = st.file_uploader(
                "Upload scanned question papers (PDF/Images)", 
                type=['pdf', 'png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="scanned_files_json"
            )
            
            if scanned_files:
                st.success(f"✅ {len(scanned_files)} file(s) uploaded")
        
        with col_process:
            st.markdown("#### ⚙️ Processing Configuration")
            
            # Get hierarchy from session if available, otherwise use empty defaults
            context = st.session_state.get('sdb_context', {})
            
            # Define common subjects for quick selection
            common_subjects = [
                "Mathematics", "Physics", "Chemistry", "Biology", 
                "Computer Science", "Data Structures", "Algorithms",
                "Machine Learning", "Database Systems", "Operating Systems",
                "Computer Networks", "Software Engineering", "Web Development",
                "Communication Skills", "English", "Technical Writing",
                "Engineering Graphics", "Engineering Mechanics", "Thermodynamics",
                "Electronics", "Digital Logic", "Microprocessors",
                "Statistics", "Probability", "Linear Algebra", "Calculus",
                "Introduction to Data Science","Communication Skills","Other (Custom)"
            ]
            
            # Subject selection with dropdown + custom option
            subject_option = st.selectbox(
                "📚 Select Subject",
                options=common_subjects,
                index=common_subjects.index("Computer Science") if "Computer Science" in common_subjects else 0,
                key="subject_dropdown"
            )
            
            # If "Other" is selected, show text input
            if subject_option == "Other (Custom)":
                process_subject = st.text_input(
                    "📚 Enter Custom Subject", 
                    value=context.get('subject', ''),
                    placeholder="Enter subject name...",
                    key="process_subject_custom"
                )
            else:
                process_subject = subject_option
                # Also show the selected subject in a disabled input for clarity
                st.text_input(
                    "📚 Selected Subject", 
                    value=process_subject,
                    disabled=True,
                    key="process_subject_display"
                )
            
            # Common colleges for quick selection
            common_colleges = [
                "VIT University", "BITS Pilani","IGDTUW", "IIT Delhi", "IIT Bombay", "IIT Madras",
                "IIT Kanpur", "IIT Kharagpur", "IIT Roorkee", "IIT Guwahati", 
                "NIT Trichy", "NIT Warangal", "IIIT Hyderabad", "DTU", "NSUT",
                "Manipal University", "SRM University", "Amrita University",
                "Other (Custom)"
            ]
            
            college_option = st.selectbox(
                "🏫 Select College",
                options=common_colleges,
                index=common_colleges.index("VIT University") if "IGDTUW" in common_colleges else 0,
                key="college_dropdown"
            )
            
            if college_option == "Other (Custom)":
                process_college = st.text_input(
                    "🏫 Enter Custom College", 
                    value=context.get('college', ''),
                    placeholder="Enter college name...",
                    key="process_college_custom"
                )
            else:
                process_college = college_option
            
            # Common branches for quick selection
            common_branches = [
                "CSE", "CSE-AI", "CSE-ML", "CSE-DS", "IT", "ECE", "EEE", "MECH", 
                "CIVIL", "CHEM", "BIO", "CSE-Cyber Security", "CSE-IoT",
                "AIDS", "AIML", "Robotics", "Biotechnology", "Aerospace",
                "Other (Custom)"
            ]
            
            branch_option = st.selectbox(
                "🌿 Select Branch",
                options=common_branches,
                index=common_branches.index("CSE-AI") if "CSE" in common_branches else 0,
                key="branch_dropdown"
            )
            
            if branch_option == "Other (Custom)":
                process_branch = st.text_input(
                    "🌿 Enter Custom Branch", 
                    value=context.get('branch', ''),
                    placeholder="Enter branch name...",
                    key="process_branch_custom"
                )
            else:
                process_branch = branch_option
            
            # Additional context fields
            col_year, col_sem = st.columns(2)
            with col_year:
                process_year = st.text_input(
                    "📅 Academic Year",
                    value=context.get('year', '2024-25'),
                    key="process_year"
                )
            
            with col_sem:
                semester_options = ["Mid-Sem", "End-Sem", "Quiz", "Assignment", "Other"]
                process_semester = st.selectbox(
                    "📖 Exam Type",
                    options=semester_options,
                    index=0,
                    key="process_semester"
                )
        
        # Validation before processing
        can_process = all([
            scanned_files,
            process_subject and process_subject.strip(),
            process_college and process_college.strip(),
            process_branch and process_branch.strip()
        ])
        
        if not can_process and scanned_files:
            st.warning("⚠️ Please fill in all required fields (Subject, College, Branch) before processing.")
        
        # Process button and JSON generation
        if can_process and st.button("🔄 Process to JSON & Load to Database", type="primary", key="process_to_json"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_questions = []
            
            for i, file in enumerate(scanned_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / (len(scanned_files) * 2))  # First half for processing
                
                try:
                    # Extract text based on file type
                    if file.type == "application/pdf":
                        extracted_text = smart_db.extract_text_from_pdf_enhanced(file)
                    else:
                        extracted_text = smart_db.extract_text_from_image(file)
                    
                    if extracted_text.strip():
                        # Parse questions using LLM
                        file_context = {
                            'subject': process_subject,
                            'college': process_college,
                            'branch': process_branch,
                            'year': process_year,
                            'semester': process_semester
                        }
                        
                        questions = smart_db.parse_questions_with_llm(extracted_text, file_context)
                        
                        # Validate and clean questions
                        for q in questions:
                            is_valid, reason = smart_db.validate_question_data(q)
                            if is_valid:
                                # Ensure all required fields are present
                                clean_question = {
                                    'question': q['question'],
                                    'subject': process_subject,
                                    'sub_topic': q.get('sub_topic', 'General'),
                                    'marks': float(q.get('marks', 1)),
                                    'year': process_year,
                                    'semester': process_semester,
                                    'branch': process_branch,
                                    'college': process_college,
                                    'unit': q.get('unit', ''),
                                    'course': context.get('course', 'B.Tech'),
                                    'verified': False
                                }
                                all_questions.append(clean_question)
                    
                except Exception as e:
                    st.warning(f"Error processing {file.name}: {e}")
            
            if all_questions:
                # Create JSON file
                status_text.text("Creating JSON file...")
                progress_bar.progress(0.6)
                
                import os
                
                # Create temporary JSON file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(all_questions, tmp_file, indent=2, ensure_ascii=False)
                    temp_json_path = tmp_file.name
                
                # Load to database using existing function
                status_text.text("Loading to database...")
                progress_bar.progress(0.8)
                
                try:
                    # Import data loader
                    from data_loader import load_pyqs_from_json
                    
                    inserted_count = load_pyqs_from_json(temp_json_path, process_subject)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Complete!")
                    
                    # Show results
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>📊 Processing Complete!</h4>
                    <ul>
                    <li>📄 Processed {len(scanned_files)} files</li>
                    <li>🔍 Extracted {len(all_questions)} questions</li>
                    <li>✅ Inserted {inserted_count} questions to database</li>
                    <li>📚 Subject: {process_subject}</li>
                    <li>🏫 College: {process_college}</li>
                    <li>🌿 Branch: {process_branch}</li>
                    <li>📅 Year: {process_year}</li>
                    <li>📖 Exam Type: {process_semester}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_json_path)
                    except:
                        pass
                    
                    # Show JSON preview
                    with st.expander("📋 Generated JSON Preview (First 2 questions)", expanded=False):
                        st.json(all_questions[:2])
                    
                    # Option to download JSON
                    json_str = json.dumps(all_questions, indent=2, ensure_ascii=False)
                    filename = f"{process_subject.lower().replace(' ', '_')}_{process_branch.lower()}_questions.json"
                    st.download_button(
                        label="📥 Download Generated JSON",
                        data=json_str,
                        file_name=filename,
                        mime="application/json",
                        key="download_generated_json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error loading to database: {e}")
                    st.info("You can still download the generated JSON file and load it manually.")
                    
                    # Still offer JSON download
                    json_str = json.dumps(all_questions, indent=2, ensure_ascii=False)
                    filename = f"{process_subject.lower().replace(' ', '_')}_{process_branch.lower()}_questions.json"
                    st.download_button(
                        label="📥 Download Generated JSON (Manual Load)",
                        data=json_str,
                        file_name=filename,
                        mime="application/json",
                        key="download_manual_json"
                    )
            
            else:
                st.error("❌ No valid questions found in uploaded files")
        
        # Manual JSON upload section
        st.markdown("---")
        st.subheader("📤 Manual JSON Upload")
        
        col_json_upload, col_json_load = st.columns([1, 1])
        
        with col_json_upload:
            uploaded_json = st.file_uploader(
                "Upload JSON file directly",
                type=['json'],
                key="manual_json_upload"
            )
            
            # Subject for manual JSON upload
            manual_subject_option = st.selectbox(
                "📚 Subject for JSON Upload",
                options=common_subjects,
                index=0,
                key="manual_json_subject_dropdown"
            )
            
            if manual_subject_option == "Other (Custom)":
                json_subject = st.text_input(
                    "📚 Enter Custom Subject for JSON",
                    value="",
                    placeholder="Enter subject name...",
                    key="json_subject_custom"
                )
            else:
                json_subject = manual_subject_option
        
        with col_json_load:
            if uploaded_json and json_subject and json_subject.strip() and st.button("📥 Load JSON to Database", key="load_manual_json"):
                try:
                    # Save uploaded JSON temporarily
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                        json_data = json.load(uploaded_json)
                        json.dump(json_data, tmp_file, indent=2)
                        temp_path = tmp_file.name
                    
                    # Load using your function
                    from data_loader import load_pyqs_from_json
                    inserted_count = load_pyqs_from_json(temp_path, json_subject)
                    
                    st.success(f"✅ Loaded {inserted_count} questions for **{json_subject}** from {uploaded_json.name}")
                    
                    # Clean up
                    import os
                    os.unlink(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error loading JSON: {e}")
        
        # Database status - COMPLETELY FIXED SECTION
        st.markdown("---")
        st.subheader("📊 Current Database Status")
        
        try:
            with SessionLocal() as db:
                # Method 1: Use ORM queries (safest approach)
                try:
                    total_questions = db.query(PYQ).count()
                    subjects_query = db.query(PYQ.subject).distinct().filter(PYQ.subject.isnot(None)).all()
                    subject_list = [s[0] for s in subjects_query] if subjects_query else []
                except Exception as orm_error:
                    # Method 2: Fallback to raw SQL with proper import
                    try:
                        total_questions = db.execute(text("SELECT COUNT(*) FROM pyqs")).scalar()
                        subjects_result = db.execute(text("SELECT DISTINCT subject FROM pyqs WHERE subject IS NOT NULL")).fetchall()
                        subject_list = [s[0] for s in subjects_result] if subjects_result else []
                    except Exception as sql_error:
                        st.error(f"Database query failed: {sql_error}")
                        total_questions = 0
                        subject_list = []
                
                # Check if columns exist
                has_frequency = hasattr(PYQ, 'frequency')
                has_branch = hasattr(PYQ, 'branch')
                has_college = hasattr(PYQ, 'college')
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Total Questions", total_questions or 0)
                with col2:
                    st.metric("📚 Subjects", len(subject_list))
                with col3:
                    if has_branch:
                        try:
                            # Try ORM first
                            branch_count = db.query(PYQ.branch).distinct().filter(PYQ.branch.isnot(None)).count()
                            st.metric("🌿 Branches", branch_count or 0)
                        except:
                            try:
                                branch_count = db.execute(text("SELECT COUNT(DISTINCT branch) FROM pyqs WHERE branch IS NOT NULL")).scalar()
                                st.metric("🌿 Branches", branch_count or 0)
                            except:
                                st.metric("🌿 Branches", "N/A")
                    else:
                        st.metric("🌿 Branches", "N/A")
                with col4:
                    if has_frequency:
                        try:
                            # Try ORM first
                            high_freq = db.query(PYQ).filter(PYQ.frequency > 2).count()
                            st.metric("🔥 High Frequency", high_freq or 0)
                        except:
                            try:
                                high_freq = db.execute(text("SELECT COUNT(*) FROM pyqs WHERE frequency > 2")).scalar()
                                st.metric("🔥 High Frequency", high_freq or 0)
                            except:
                                st.metric("🔥 High Frequency", "N/A")
                    else:
                        st.metric("🔥 High Frequency", "N/A")
                
                # Subject-wise breakdown with better formatting
                if subject_list:
                    st.subheader("📊 Subject-wise Distribution")
                    
                    # Create a more organized display
                    subject_data = []
                    for subject in subject_list:
                        try:
                            # Try ORM first
                            count = db.query(PYQ).filter(PYQ.subject == subject).count()
                            subject_data.append((subject, count))
                        except:
                            try:
                                count = db.execute(text("SELECT COUNT(*) FROM pyqs WHERE subject = :subject"), 
                                                {"subject": subject}).scalar()
                                subject_data.append((subject, count))
                            except:
                                subject_data.append((subject, "Error"))
                    
                    # Sort by count (descending)
                    subject_data.sort(key=lambda x: x[1] if isinstance(x[1], int) else 0, reverse=True)
                    
                    # Display in columns for better layout
                    cols = st.columns(3)
                    for i, (subject, count) in enumerate(subject_data):
                        with cols[i % 3]:
                            if isinstance(count, int):
                                st.metric(f"📚 {subject}", count)
                            else:
                                st.write(f"📚 **{subject}**: {count}")
                
                # Recent additions
                st.subheader("📅 Recent Questions")
                try:
                    # Try ORM first
                    recent_questions = (db.query(PYQ.question, PYQ.subject, PYQ.sub_topic, PYQ.marks)
                                    .order_by(PYQ.id.desc())
                                    .limit(5)
                                    .all())
                except:
                    try:
                        recent_questions = db.execute(text(
                            "SELECT question, subject, sub_topic, marks FROM pyqs ORDER BY id DESC LIMIT 5"
                        )).fetchall()
                    except:
                        recent_questions = []
                        st.warning("Could not retrieve recent questions")
                
                for q in recent_questions:
                    question_text = q[0] or "No question text"
                    subject_name = q[1] or "No subject"
                    sub_topic = q[2] or "N/A"
                    marks = q[3] or "N/A"
                    
                    # Truncate question text for display
                    display_text = question_text[:100] + "..." if len(question_text) > 100 else question_text
                    
                    with st.expander(f"📚 {subject_name} | {display_text}"):
                        st.write(f"**Subject:** {subject_name}")
                        st.write(f"**Sub-topic:** {sub_topic}")
                        st.write(f"**Marks:** {marks}")
                        st.write(f"**Full Question:** {question_text}")
        
        except Exception as e:
            st.error(f"Database connection error: {e}")
            # Add debug information
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
with tab3:

    st.header("🔍Search Engine")
    
    try:
        with SessionLocal() as db:

            
            col1, col2, col3, col4 = st.columns(4)
            
            search_term = st.text_input("Enter search term:", key="search_questions")

            try:
                # Get subjects list safely
                subjects_query = db.execute(text("SELECT DISTINCT subject FROM pyqs WHERE subject IS NOT NULL")).fetchall()
                subject_options = ["All"] + [row[0] for row in subjects_query]
                
                search_subject = st.selectbox(
                    "Filter by Subject:", 
                    subject_options,
                    key="search_subject_filter"
                )
                
                if st.button("🔍 Search", key="perform_search"):
                    if search_term.strip():  # Check for non-empty search term
                        # Initialize query and params variables
                        query = ""
                        params = {}
                        
                        if search_subject != "All":
                            # Search with subject filter
                            query = """
                            SELECT question, subject, COALESCE(sub_topic, 'N/A'), COALESCE(marks, 0) 
                            FROM pyqs 
                            WHERE LOWER(COALESCE(question, '')) LIKE LOWER(:search_term) 
                            AND subject = :subject
                            LIMIT 10
                            """
                            params = {
                                'search_term': f"%{search_term}%",
                                'subject': search_subject
                            }
                        else:
                            # Search without subject filter
                            query = """
                            SELECT question, subject, COALESCE(sub_topic, 'N/A'), COALESCE(marks, 0) 
                            FROM pyqs 
                            WHERE LOWER(COALESCE(question, '')) LIKE LOWER(:search_term)
                            LIMIT 10
                            """
                            params = {
                                'search_term': f"%{search_term}%"
                            }
                        
                        # Execute the query
                        results = db.execute(text(query), params).fetchall()
                        
                        if results:
                            st.write(f"Found {len(results)} results:")
                            for i, result in enumerate(results):
                                question = result[0] or "No question text"
                                subject = result[1] or "No subject"
                                sub_topic = result[2] or "N/A"
                                marks = result[3] or "0"
                                
                                question_preview = question[:100] + "..." if len(question) > 100 else question
                                with st.expander(f"Result {i+1}: {question_preview}"):
                                    st.write(f"**Subject:** {subject}")
                                    st.write(f"**Sub-topic:** {sub_topic}")
                                    st.write(f"**Marks:** {marks}")
                                    st.write(f"**Question:** {question}")
                        else:
                            st.info("No questions found matching your search criteria.")
                    else:
                        st.warning("Please enter a search term.")
                        
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                # Add debug information with proper variable checking
                query_info = query if 'query' in locals() and query else 'Not initialized'
                params_info = params if 'params' in locals() and params else 'Not initialized'
                st.error(f"Debug info: Query: {query_info}, Params: {params_info}")

    except Exception as e:
        st.error(f"Analytics error: {str(e)}")
        st.info("Check your database connection and ensure the 'pyqs' table exists with the expected structure.")


# TAB 4: Study Planner
with tab4:
    
    st.markdown("""
        <div class="smart-db-header">
            <h2>📅 Study Planner</h2>
            <p>Create personalized study plans based on PYQ frequency and difficulty</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Study Planner Configuration
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🎯 Study Plan Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        plan_college = st.selectbox(
            "🏫 College",
            ["All Colleges"] + list(UNIVERSITY_CONFIG.keys()) + ["Other"],
            key="plan_college"
        )
        
        plan_subject = st.selectbox(
            "📚 Subject",
            ["Data Structures", "Computer Networks", "Database Management", 
             "Operating Systems", "Software Engineering", "Machine Learning",
             "Artificial Intelligence", "Cyber Security", "Other"],
            key="plan_subject"
        )
    
    with col2:
        plan_branch = st.selectbox(
            "🌿 Branch",
            get_available_branches(plan_college),
            key="plan_branch"
        )
        
        plan_duration = st.selectbox(
            "⏱️ Study Duration",
            [7, 14, 21, 30, 45, 60],
            format_func=lambda x: f"{x} days",
            index=3,  # Default to 30 days
            key="plan_duration"
        )
    
    with col3:
        plan_semester = st.selectbox(
            "📝 Semester Type",
            ["All Types", "MID SEMESTER", "END SEMESTER"],
            key="plan_semester"
        )
        
        questions_per_day = st.slider(
            "❓ Target Questions/Day",
            min_value=3,
            max_value=20,
            value=8,
            key="questions_per_day"
        )
    
    # Create Study Plan
    if st.button("📋 Create Study Plan", type="primary", key="create_study_plan"):
        context = {
            'college': plan_college if plan_college != "All Colleges" else None,
            'branch': plan_branch if plan_branch != "All Branches" else None,
            'subject': plan_subject,
            'semester': plan_semester if plan_semester != "All Types" else None
        }
        
        # Remove None values
        context = {k: v for k, v in context.items() if v is not None}
        
        with st.spinner("Creating personalized study plan..."):
            try:
                with SessionLocal() as session:
                    study_plan = study_planner.create_study_plan(
                        session, context, plan_duration
                    )
                
                if 'error' in study_plan:
                    st.error(f"❌ {study_plan['error']}")
                else:
                    st.session_state.study_plan = study_plan
                    st.success("✅ Study plan created successfully!")
            
            except Exception as e:
                st.error(f"❌ Error creating study plan: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display Study Plan
    if st.session_state.study_plan:
        plan = st.session_state.study_plan
        
        # Study Plan Overview
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.header("📊 Study Plan Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Duration", f"{plan['duration_days']} days")
        with col2:
            st.metric("❓ Total Questions", plan['total_questions'])
        with col3:
            st.metric("📖 Questions/Day", plan['questions_per_day'])
        with col4:
            st.metric("⏰ Est. Time", f"{plan['total_estimated_hours']:.1f} hours")
        
        # Progress Tracking
        st.subheader("📈 Progress Tracking")
        
        completed_days = st.slider(
            "Mark completed days:",
            min_value=0,
            max_value=plan['duration_days'],
            value=st.session_state.study_progress_days,
            key="progress_slider"
        )
        
        if completed_days != st.session_state.study_progress_days:
            st.session_state.study_progress_days = completed_days
        
        # Calculate and display progress
        progress = study_planner.get_study_progress(plan, completed_days)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "📅 Day Progress",
                f"{progress['completed_days']}/{progress['total_days']}",
                f"{progress['completion_percentage']:.1f}%"
            )
        
        with col2:
            st.metric(
                "❓ Question Progress",
                f"{progress['questions_completed']}/{progress['total_questions']}",
                f"{progress['question_progress_percentage']:.1f}%"
            )
        
        # Progress bars
        st.progress(progress['completion_percentage'] / 100)
        st.caption(f"Time spent: {progress['time_spent_minutes']/60:.1f} hours / {progress['total_estimated_minutes']/60:.1f} hours")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Topic Distribution
        if 'topic_distribution' in plan and plan['topic_distribution']:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📚 Topic Distribution")
            
            topics = list(plan['topic_distribution'].keys())
            counts = list(plan['topic_distribution'].values())
            
            # Display as columns for better visualization
            num_topics = len(topics)
            cols_per_row = 3
            
            for i in range(0, num_topics, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < num_topics:
                        topic = topics[i + j]
                        count = counts[i + j]
                        with col:
                            st.metric(f"📖 {topic}", f"{count} questions")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Daily Schedule
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📅 Daily Study Schedule")
        
        # Filter for current week or show all
        show_option = st.radio(
            "Display options:",
            ["Current Week", "All Days", "Upcoming Days"],
            horizontal=True,
            key="schedule_display"
        )
        
        daily_targets = plan['daily_targets']
        
        if show_option == "Current Week":
            # Show next 7 days from current progress
            start_day = max(0, completed_days)
            end_day = min(len(daily_targets), start_day + 7)
            display_targets = daily_targets[start_day:end_day]
        elif show_option == "Upcoming Days":
            # Show only upcoming days
            display_targets = daily_targets[completed_days:]
        else:
            # Show all days
            display_targets = daily_targets
        
        for target in display_targets:
            day_num = target['day']
            is_completed = day_num <= completed_days
            
            # Style based on completion status
            if is_completed:
                status_emoji = "✅"
                card_style = "background-color: #d4edda; border-left: 4px solid #28a745;"
            elif day_num == completed_days + 1:
                status_emoji = "📍"
                card_style = "background-color: #fff3cd; border-left: 4px solid #ffc107;"
            else:
                status_emoji = "⏳"
                card_style = "background-color: #f8f9fa; border-left: 4px solid #6c757d;"
            
            with st.expander(
                f"{status_emoji} Day {day_num}: {target['primary_topic']} "
                f"({target['questions_count']} questions, {target['estimated_time_minutes']} min)",
                expanded=(day_num == completed_days + 1)  # Expand next day
            ):
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 5px; {card_style}">
                <h4 style="color: black;>📚 Primary Topic: {target['primary_topic']}</h4>
                <p style="color: black;><strong>Questions:</strong> {target['questions_count']} | 
                   <strong>Total Marks:</strong> {target['total_marks']} | 
                   <strong>Est. Time:</strong> {target['estimated_time_minutes']} minutes</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show questions for the day
                st.markdown("**Questions for today:**")
                for i, question in enumerate(target['questions'][:3]):  # Show first 3
                    st.markdown(f"""
                    <div class="question-card">
                    <strong>Q{i+1}:</strong> {question['question']}<br>
                    <small>Marks: {question['marks']} | Year: {question['year']} | Sub-topic: {question['sub_topic']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(target['questions']) > 3:
                    st.info(f"... and {len(target['questions']) - 3} more questions")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export Options
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📤 Export Study Plan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report", key="export_pdf"):
                st.info("📄 PDF export functionality can be implemented")
        
        with col2:
            if st.button("📧 Email Schedule", key="email_schedule"):
                st.info("📧 Email functionality can be implemented")
        
        with col3:
            if st.button("📱 Mobile Reminder", key="mobile_reminder"):
                st.info("📱 Mobile reminder functionality can be implemented")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show sample/demo when no plan exists
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.info("📋 Create a study plan above to see your personalized schedule, progress tracking, and topic distribution.")
        
        # Sample preview
        st.subheader("🎯 Study Planner Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Plan Features:**
            - Personalized daily schedules
            - Progress tracking
            - Time estimation
            - Topic-wise distribution
            - Difficulty-based ordering
            """)
        
        with col2:
            st.markdown("""
            **📈 Progress Tracking:**
            - Daily completion status
            - Visual progress bars
            - Time spent tracking
            - Performance analytics
            - Motivational milestones
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
# Footer
