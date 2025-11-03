import streamlit as st
import tempfile
import json
import hashlib
import re
import os
from typing import Dict, List, Tuple
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from sqlalchemy.orm import sessionmaker
from database import SessionLocal, PYQ, FileRecord
from langchain.chat_models import ChatOpenAI

# Try to import optional dependencies
try:
    import pytesseract
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def generate_question_hash(question: str) -> str:
    """Generate a hash for question deduplication"""
    if not question or not question.strip():
        return ""
    
    # Normalize the question text
    normalized = re.sub(r'\s+', ' ', question.lower().strip())
    normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def check_question_exists_by_hash(question_hash: str, subject: str = None) -> bool:
    """Check if question already exists in database by hash"""
    if not question_hash:
        return False
    
    try:
        with SessionLocal() as db:
            query = db.query(PYQ).filter(PYQ.question_hash == question_hash)
            if subject:
                query = query.filter(PYQ.subject == subject)
            existing = query.first()
            return existing is not None
    except Exception as e:
        print(f"Error checking question existence: {e}")
        return False

def save_questions_to_database_with_hash(questions: List[Dict], context: Dict) -> Tuple[int, int, int]:
    """Save questions to database with hash generation and deduplication"""
    added_count = 0
    updated_count = 0
    skipped_count = 0
    
    try:
        with SessionLocal() as db:
            for q_data in questions:
                try:
                    # Generate question hash
                    question_text = q_data.get('question', '')
                    question_hash = generate_question_hash(question_text)
                    
                    if not question_hash:
                        skipped_count += 1
                        continue
                    
                    # Check if question already exists
                    existing = db.query(PYQ).filter(PYQ.question_hash == question_hash).first()
                    
                    if existing:
                        # Update frequency if question exists
                        if hasattr(existing, 'frequency'):
                            existing.frequency = getattr(existing, 'frequency', 1) + 1
                        else:
                            # Add frequency field if it doesn't exist
                            existing.frequency = 2
                        
                        # Update other fields if needed
                        if hasattr(existing, 'year') and context.get('year'):
                            current_years = str(existing.year).split(',') if existing.year else []
                            new_year = str(context['year'])
                            if new_year not in current_years:
                                existing.year = f"{existing.year},{new_year}" if existing.year else new_year
                        
                        updated_count += 1
                        
                    else:
                        # Create new question with hash
                        question_data = {
                            'question': question_text,
                            'question_hash': question_hash,
                            'subject': context.get('subject', q_data.get('subject', '')),
                            'sub_topic': q_data.get('sub_topic', 'General'),
                            'marks': float(q_data.get('marks', 1)),
                            'year': context.get('year', ''),
                            'semester': context.get('semester', ''),
                            'branch': context.get('branch', ''),
                            'college': context.get('college', ''),
                            'unit': q_data.get('unit', ''),
                            'course': context.get('course', 'B.Tech'),
                            'verified': False,
                            'frequency': 1
                        }
                        
                        new_question = PYQ(**question_data)
                        db.add(new_question)
                        added_count += 1
                
                except Exception as e:
                    print(f"Error processing question: {e}")
                    skipped_count += 1
                    continue
            
            db.commit()
            
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
        return 0, 0, len(questions)
    
    return added_count, updated_count, skipped_count

class EnhancedSmartDatabaseBuilder:
    def __init__(self):
        self.processed_files = set()  # Track processed files in session
        self.session_questions = []   # Track questions in current session
    
    def check_file_processed(self, filename: str) -> bool:
        """Check if file has been processed before"""
        try:
            with SessionLocal() as session:
                existing = session.query(FileRecord).filter(
                    FileRecord.filename == filename,
                    FileRecord.processing_status == "completed"
                ).first()
                return existing is not None
        except Exception as e:
            print(f"Error checking file processing status: {e}")
            return False
    
    def record_file_processing(self, filename: str, context: Dict, questions_count: int):
        """Record that a file has been processed"""
        try:
            with SessionLocal() as session:
                file_record = FileRecord(
                    filename=filename,
                    file_type="question_paper",
                    college=context.get('college', 'Unknown'),
                    course=context.get('course', 'Unknown'),
                    subject=context.get('subject', 'Unknown'),
                    questions_extracted=questions_count,
                    processing_status="completed",
                    processed_at=datetime.utcnow()
                )
                session.add(file_record)
                session.commit()
        except Exception as e:
            print(f"Error recording file processing: {e}")
    
    def preprocess_image_for_ocr(self, image):
        """Enhanced image preprocessing for better OCR results"""
        if not OCR_AVAILABLE:
            raise Exception("OCR libraries not available")
            
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
        if not OCR_AVAILABLE:
            st.error("OCR not available - required libraries not installed")
            return ""
            
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
                
                if len(text.strip()) < 50 and OCR_AVAILABLE:  # If little text found, use OCR
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
        question_indicators = ['what', 'how', 'why', 'explain', 'define', 'solve', 'find', 
                             'prove', 'derive', 'calculate', 'evaluate', 'determine', 
                             'show', 'verify', 'trace', 'compute']
        
        if not any(indicator in question_text for indicator in question_indicators):
            return False, "Text doesn't appear to be a question"
        
        # Avoid headers and instructions
        avoid_terms = ['time:', 'marks:', 'instructions:', 'note:', 'attempt']
        if any(term in question_text for term in avoid_terms):
            return False, "Appears to be instruction or header text"
        
        return True, "Valid question"
    
    def process_files_to_json_with_dedup(self, files, context: Dict) -> Tuple[List[Dict], Dict]:
        """Process files to JSON with comprehensive deduplication"""
        all_questions = []
        processing_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'questions_extracted': 0,
            'duplicates_skipped': 0,
            'new_questions': 0
        }
        
        for file in files:
            filename = file.name
            
            # Check if file was already processed
            if self.check_file_processed(filename):
                st.info(f"File {filename} already processed, skipping...")
                processing_stats['files_skipped'] += 1
                continue
            
            # Check if file was processed in current session
            if filename in self.processed_files:
                st.info(f"File {filename} already processed in this session, skipping...")
                processing_stats['files_skipped'] += 1
                continue
            
            # Extract text based on file type
            if file.type == "application/pdf":
                extracted_text = self.extract_text_from_pdf_enhanced(file)
            else:
                extracted_text = self.extract_text_from_image(file)
            
            if not extracted_text.strip():
                st.warning(f"No text extracted from {filename}")
                continue
            
            # Parse questions using LLM
            questions = self.parse_questions_with_llm(extracted_text, context)
            
            # Validate and deduplicate questions
            file_questions = []
            for q in questions:
                is_valid, reason = self.validate_question_data(q)
                if not is_valid:
                    continue
                
                # Generate hash and check for duplicates
                question_hash = generate_question_hash(q['question'])
                
                # Check database for existing question
                if check_question_exists_by_hash(question_hash):
                    processing_stats['duplicates_skipped'] += 1
                    continue
                
                # Check current session for duplicates
                if question_hash in [generate_question_hash(sq['question']) for sq in self.session_questions]:
                    processing_stats['duplicates_skipped'] += 1
                    continue
                
                # Add context and hash to question
                clean_question = {
                    'question': q['question'],
                    'question_hash': question_hash,
                    'subject': context['subject'],
                    'sub_topic': q.get('sub_topic', 'General'),
                    'marks': float(q.get('marks', 1)),
                    'year': context['year'],
                    'semester': context['semester'],
                    'branch': context['branch'],
                    'college': context['college'],
                    'unit': q.get('unit', ''),
                    'course': context.get('course', 'B.Tech'),
                    'verified': False,
                    'frequency': 1
                }
                
                file_questions.append(clean_question)
                self.session_questions.append(clean_question)
                processing_stats['new_questions'] += 1
            
            all_questions.extend(file_questions)
            processing_stats['questions_extracted'] += len(file_questions)
            processing_stats['files_processed'] += 1
            
            # Mark file as processed in current session
            self.processed_files.add(filename)
            
            # Record file processing in database
            self.record_file_processing(filename, context, len(file_questions))
        
        return all_questions, processing_stats
    
    def export_to_json(self, questions: List[Dict], filename: str) -> str:
        """Export questions to JSON string"""
        try:
            # Remove database-specific fields for clean JSON
            clean_questions = []
            for q in questions:
                clean_q = {k: v for k, v in q.items() if k not in ['question_hash', 'id', 'created_at', 'updated_at']}
                clean_questions.append(clean_q)
            
            return json.dumps(clean_questions, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error creating JSON: {e}")
            return ""
    
    def clear_session_data(self):
        """Clear session tracking data"""
        self.processed_files.clear()
        self.session_questions.clear()

def enhanced_smart_database_builder_interface():
    """Enhanced Smart Database Builder interface with deduplication"""
    
    if 'enhanced_sdb' not in st.session_state:
        st.session_state.enhanced_sdb = EnhancedSmartDatabaseBuilder()
    
    sdb = st.session_state.enhanced_sdb
    
    st.markdown("### Enhanced Smart Database Builder with Deduplication")
    
    # Check dependencies
    if not OCR_AVAILABLE:
        st.warning("OCR capabilities limited - install pytesseract, opencv-python, and numpy for full functionality")
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        return sdb
    
    # Session management
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Session status: {len(sdb.processed_files)} files processed, {len(sdb.session_questions)} questions in memory")
    with col2:
        if st.button("Clear Session", key="clear_sdb_session"):
            sdb.clear_session_data()
            st.success("Session data cleared. You can now process files again.")
            st.rerun()
    
    # Context form
    st.markdown("#### Context Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        college = st.selectbox(
            "College",
            ["Select College", "NSUT", "DTU", "IGDTUW", "Other"],
            key="enhanced_sdb_college"
        )
        
        course = st.selectbox(
            "Course",
            ["B.Tech", "BE", "B.Com", "BDS", "M.Tech", "Other"],
            key="enhanced_sdb_course"
        )
        
        branch = st.text_input("Branch", key="enhanced_sdb_branch")
    
    with col2:
        subject = st.text_input("Subject", key="enhanced_sdb_subject")
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2024, key="enhanced_sdb_year")
        semester = st.selectbox(
            "Semester Type",
            ["MID SEMESTER", "END SEMESTER", "QUIZ", "ASSIGNMENT"],
            key="enhanced_sdb_semester"
        )
    
    # File upload
    st.markdown("#### File Upload")
    files = st.file_uploader(
        "Upload Question Papers",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="enhanced_sdb_files"
    )
    
    if files and subject and college != "Select College" and branch:
        context = {
            'college': college,
            'course': course,
            'branch': branch,
            'subject': subject,
            'year': str(year),
            'semester': semester
        }
        
        # Show context
        with st.expander("Processing Context", expanded=True):
            for key, value in context.items():
                st.write(f"**{key.title()}:** {value}")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("Process Files", type="primary", key="enhanced_process")
        with col2:
            export_json = st.checkbox("Export to JSON", key="enhanced_export_json")
        
        if process_button:
            with st.spinner("Processing files..."):
                questions, stats = sdb.process_files_to_json_with_dedup(files, context)
                
                # Show processing statistics
                st.markdown("#### Processing Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Files Processed", stats['files_processed'])
                with col2:
                    st.metric("Files Skipped", stats['files_skipped'])
                with col3:
                    st.metric("Questions Extracted", stats['questions_extracted'])
                with col4:
                    st.metric("Duplicates Skipped", stats['duplicates_skipped'])
                
                if questions:
                    # Save to database
                    with st.spinner("Saving to database..."):
                        added, updated, skipped = save_questions_to_database_with_hash(questions, context)
                    
                    st.success("Database update completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Added", added)
                    with col2:
                        st.metric("Updated", updated)
                    with col3:
                        st.metric("Skipped", skipped)
                    
                    # Show sample questions
                    if questions:
                        with st.expander("Sample Extracted Questions", expanded=False):
                            for i, q in enumerate(questions[:5]):
                                st.markdown(f"**Q{i+1}:** {q['question']}")
                                st.markdown(f"*Marks:* {q.get('marks', 1)} | *Sub-topic:* {q.get('sub_topic', 'General')} | *Unit:* {q.get('unit', 'N/A')}")
                                st.markdown("---")
                    
                    # Export to JSON if requested
                    if export_json and questions:
                        json_str = sdb.export_to_json(questions, f"{subject}_{year}_questions.json")
                        if json_str:
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"{subject}_{year}_questions.json",
                                mime="application/json"
                            )
                else:
                    st.warning("No valid questions were extracted from the uploaded files.")
    
    elif files:
        missing_fields = []
        if college == "Select College":
            missing_fields.append("College")
        if not subject:
            missing_fields.append("Subject")
        if not branch:
            missing_fields.append("Branch")
        
        st.warning(f"Please fill in the following required fields: {', '.join(missing_fields)}")
    
    return sdb