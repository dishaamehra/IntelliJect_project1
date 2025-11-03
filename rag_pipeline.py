import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sqlalchemy.orm import Session
from database import PYQ
from sqlalchemy import text

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = OpenAI()
# Initialize embeddings
embedding = OpenAIEmbeddings()

def load_vectorstore_from_db(session: Session, subject: str = None) -> FAISS:
    """
    Dynamically builds a FAISS vectorstore from PYQs stored in the database for the given subject.
    """
    query_set = session.query(PYQ)
    if subject:
        query_set = query_set.filter(PYQ.subject == subject)

    pyqs = query_set.all()
    if not pyqs:
        return None  # No data to build vector store
    
    docs = [
        Document(
            page_content=pyq.question,
            metadata={
                "year": pyq.year,
                "subject": pyq.subject,
                "sub_topic": pyq.sub_topic,
                "marks": pyq.marks,
                "frequency": getattr(pyq, 'frequency', 1),
                "unit": getattr(pyq, 'unit', ''),
                "semester": getattr(pyq, 'semester', ''),
                "branch": getattr(pyq, 'branch', ''),
                "college": getattr(pyq, 'college', ''),
                "course": getattr(pyq, 'course', '')
            }
        )
        for pyq in pyqs
    ]
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore

def semantic_search_db(session: Session, query: str, subject: str = None, k: int = 5) -> List[Document]:
    """
    Perform semantic search over PYQs stored in the DB using FAISS.
    """
    vectorstore = load_vectorstore_from_db(session, subject)
    if not vectorstore:
        return []

    # Use max marginal relevance search for better diversity
    results = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=20)
    return results

def infer_subtopic(text: str) -> str:
    """
    Infer subtopic using OpenAI API directly.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying academic subtopics from text content."
                },
                {
                    "role": "user", 
                    "content": f"""Read the following academic content and suggest the most relevant subtopic (like 'Firewall', 'Water Pollution', etc.) in 2-3 words:

{text}

Subtopic:"""
                }
            ],
            max_tokens=50,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Subtopic inference failed:", e)
        return "General"

def get_relevant_pyqs(session: Session, query: str, subject: str = None, k: int = 3) -> List[Document]:
    """
    Get relevant PYQs from database using semantic similarity search only (no JSON fallback).
    """
    return semantic_search_db(session, query, subject, k)

def nlp_chunk_text(text: str, max_sentences: int = 5) -> List[str]:
    """
    Simple text chunking by sentence count.
    Make sure to call nltk.download('punkt') once in your environment/setup.
    """
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Download punkt if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = sent_tokenize(text)
        chunks = []

        for i in range(0, len(sentences), max_sentences):
            chunk = ' '.join(sentences[i:i + max_sentences])
            chunks.append(chunk)

        return chunks
    except Exception as e:
        print(f"Error in text chunking: {e}")
        # Fallback to simple splitting
        sentences = text.split('. ')
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = '. '.join(sentences[i:i + max_sentences])
            chunks.append(chunk)
        return chunks

def process_notes_and_match_pyqs(text: str, subject: str, session: Session, k: int = 3):
    """
    Processes the notes text: chunk, infer subtopic, and get matching PYQs from DB.
    """
    chunks = nlp_chunk_text(text)
    results = []
    
    for chunk in chunks:
        # Get matches first
        matches = get_relevant_pyqs(session, chunk, subject, k=k)
        
        # Then extract subtopic from database results or infer it
        if matches:
            subtopic = matches[0].metadata.get('sub_topic', 'General')
            if not subtopic or subtopic.strip() == '':
                subtopic = infer_subtopic(chunk)
        else:
            subtopic = "No matches found"
            
        results.append({
            "chunk": chunk,
            "subtopic": subtopic,
            "matches": matches
        })
    
    return results

def enhanced_semantic_search(session: Session, query: str, subject: str = None, 
                            filters: dict = None, k: int = 5) -> List[Document]:
    """
    Enhanced semantic search with additional filtering capabilities.
    
    Args:
        session: Database session
        query: Search query
        subject: Subject filter
        filters: Additional filters (year, semester, branch, etc.)
        k: Number of results to return
    """
    # Build base query
    query_set = session.query(PYQ)
    
    if subject:
        query_set = query_set.filter(PYQ.subject == subject)
    
    # Apply additional filters
    if filters:
        if filters.get('year'):
            query_set = query_set.filter(PYQ.year.like(f"%{filters['year']}%"))
        if filters.get('semester'):
            query_set = query_set.filter(PYQ.semester == filters['semester'])
        if filters.get('branch'):
            query_set = query_set.filter(PYQ.branch == filters['branch'])
        if filters.get('min_marks'):
            query_set = query_set.filter(PYQ.marks >= filters['min_marks'])
        if filters.get('min_frequency'):
            query_set = query_set.filter(PYQ.frequency >= filters['min_frequency'])
    
    pyqs = query_set.all()
    
    if not pyqs:
        return []
    
    # Create documents for vectorstore
    docs = [
        Document(
            page_content=pyq.question,
            metadata={
                "id": pyq.id,
                "year": pyq.year,
                "subject": pyq.subject,
                "sub_topic": pyq.sub_topic,
                "marks": pyq.marks,
                "frequency": getattr(pyq, 'frequency', 1),
                "unit": getattr(pyq, 'unit', ''),
                "semester": getattr(pyq, 'semester', ''),
                "branch": getattr(pyq, 'branch', ''),
                "college": getattr(pyq, 'college', ''),
                "course": getattr(pyq, 'course', '')
            }
        )
        for pyq in pyqs
    ]
    
    # Build vectorstore and search
    vectorstore = FAISS.from_documents(docs, embedding)
    results = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=min(len(docs), 20))
    
    return results

def get_question_recommendations(session: Session, topic: str, subject: str = None, 
                               difficulty_level: str = None, k: int = 5) -> List[Document]:
    """
    Get question recommendations based on topic and difficulty level.
    
    Args:
        session: Database session
        topic: Topic/subtopic to search for
        subject: Subject filter
        difficulty_level: 'easy', 'medium', 'hard' based on marks
        k: Number of recommendations
    """
    filters = {}
    
    # Map difficulty to marks range
    if difficulty_level:
        if difficulty_level.lower() == 'easy':
            filters['max_marks'] = 5
        elif difficulty_level.lower() == 'medium':
            filters['min_marks'] = 5
            filters['max_marks'] = 15
        elif difficulty_level.lower() == 'hard':
            filters['min_marks'] = 15
    
    query_set = session.query(PYQ)
    
    if subject:
        query_set = query_set.filter(PYQ.subject == subject)
    
    # Filter by topic in question text or sub_topic
    query_set = query_set.filter(
        (PYQ.question.ilike(f"%{topic}%")) |
        (PYQ.sub_topic.ilike(f"%{topic}%"))
    )
    
    # Apply difficulty filters
    if filters.get('min_marks'):
        query_set = query_set.filter(PYQ.marks >= filters['min_marks'])
    if filters.get('max_marks'):
        query_set = query_set.filter(PYQ.marks <= filters['max_marks'])
    
    # Order by frequency (more frequently asked questions first)
    query_set = query_set.order_by(PYQ.frequency.desc())
    
    pyqs = query_set.limit(k * 2).all()  # Get more for semantic filtering
    
    if not pyqs:
        return []
    
    # Convert to documents and perform semantic search
    docs = [
        Document(
            page_content=pyq.question,
            metadata={
                "id": pyq.id,
                "year": pyq.year,
                "subject": pyq.subject,
                "sub_topic": pyq.sub_topic,
                "marks": pyq.marks,
                "frequency": getattr(pyq, 'frequency', 1),
                "unit": getattr(pyq, 'unit', ''),
                "difficulty": difficulty_level or "unknown"
            }
        )
        for pyq in pyqs
    ]
    
    if len(docs) <= k:
        return docs
    
    # Use semantic search to get the most relevant ones
    vectorstore = FAISS.from_documents(docs, embedding)
    results = vectorstore.similarity_search(topic, k=k)
    
    return results

def generate_study_plan(session: Session, subject: str, topics: List[str]) -> dict:
    """
    Generate a study plan based on available questions for given topics.
    
    Args:
        session: Database session
        subject: Subject name
        topics: List of topics to cover
    
    Returns:
        dict: Study plan with topics, question counts, and recommendations
    """
    study_plan = {
        'subject': subject,
        'topics': [],
        'total_questions': 0,
        'high_frequency_topics': [],
        'recommendations': []
    }
    
    for topic in topics:
        # Get questions for this topic
        questions = get_question_recommendations(session, topic, subject, k=10)
        
        if questions:
            topic_info = {
                'topic': topic,
                'question_count': len(questions),
                'avg_marks': sum([q.metadata.get('marks', 0) for q in questions]) / len(questions),
                'max_frequency': max([q.metadata.get('frequency', 1) for q in questions]),
                'sample_questions': [q.page_content[:100] + "..." for q in questions[:3]]
            }
            
            study_plan['topics'].append(topic_info)
            study_plan['total_questions'] += len(questions)
            
            # Mark high frequency topics
            if topic_info['max_frequency'] > 2:
                study_plan['high_frequency_topics'].append(topic)
    
    # Generate recommendations
    if study_plan['high_frequency_topics']:
        study_plan['recommendations'].append(
            f"Focus on high-frequency topics: {', '.join(study_plan['high_frequency_topics'])}"
        )
    
    # Sort topics by importance (frequency and question count)
    study_plan['topics'].sort(key=lambda x: (x['max_frequency'], x['question_count']), reverse=True)
    
    return study_plan

# Utility functions for enhanced RAG
def extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts from text using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract key academic concepts, topics, and important terms from the given text. Return them as a comma-separated list."
                },
                {
                    "role": "user",
                    "content": f"Extract key concepts from: {text[:1000]}..."
                }
            ],
            max_tokens=200,
            temperature=0
        )
        
        concepts = response.choices[0].message.content.strip()
        return [concept.strip() for concept in concepts.split(',')]
    
    except Exception as e:
        print(f"Error extracting concepts: {e}")
        return []

def similarity_score(query: str, document: str) -> float:
    """
    Calculate similarity score between query and document using embeddings.
    """
    try:
        # Get embeddings for both texts
        query_embedding = embedding.embed_query(query)
        doc_embedding = embedding.embed_query(document)
        
        # Calculate cosine similarity
        import numpy as np
        
        query_vec = np.array(query_embedding)
        doc_vec = np.array(doc_embedding)
        
        # Cosine similarity
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        
        return float(similarity)
    
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0


def load_vectorstore_from_db_with_filters(session: Session, filter_params: dict = None) -> FAISS:
    """
    Dynamically builds a FAISS vectorstore from PYQs with applied filters.
    """
    query_set = session.query(PYQ)
    
    # Apply filters if provided
    if filter_params:
        for field, value in filter_params.items():
            if hasattr(PYQ, field):
                query_set = query_set.filter(getattr(PYQ, field) == value)

    pyqs = query_set.all()
    if not pyqs:
        return None  # No data to build vector store
        
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
                "course": getattr(pyq, 'course', None),
                "frequency": getattr(pyq, 'frequency', 1),
            }
        )
        for pyq in pyqs
    ]
    
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore


def semantic_search_db_with_filters(session: Session, query: str, filter_params: dict = None, k: int = 5) -> List[Document]:
    """
    Perform semantic search over filtered PYQs stored in the DB using FAISS.
    """
    vectorstore = load_vectorstore_from_db_with_filters(session, filter_params)
    if not vectorstore:
        return []

    results = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=20)
    return results


def get_relevant_pyqs_with_filters(session: Session, query: str, filter_params: dict = None, k: int = 3) -> List[Document]:
    """
    Get relevant PYQs from database using semantic similarity search with filters applied.
    """
    return semantic_search_db_with_filters(session, query, filter_params, k)


def process_notes_and_match_pyqs_with_filters(text: str, filter_params: dict, session: Session, k: int = 3):
    """
    Enhanced version of process_notes_and_match_pyqs with filter support.
    """
    chunks = nlp_chunk_text(text)
    results = []
    
    for chunk in chunks:
        # Get matches with filters applied
        matches = get_relevant_pyqs_with_filters(session, chunk, filter_params, k=k)
        if not matches:
            # Skip chunks without results
            continue
        # Extract subtopic from database results
        if matches:
            subtopic = matches[0].metadata.get('sub_topic', 'General')
        else:
            subtopic = "No matches found"
            
        results.append({
            "chunk": chunk,
            "subtopic": subtopic,
            "matches": matches,
            "applied_filters": filter_params
        })
    
    return results


def get_database_filter_options(session: Session) -> dict:
    """
    Get available filter options from the database for dynamic dropdown population.
    """
    try:
        filter_options = {}
        
        # Get subjects
        subjects = session.execute(text("SELECT DISTINCT subject FROM pyqs WHERE subject IS NOT NULL")).fetchall()
        filter_options['subjects'] = [s[0] for s in subjects] if subjects else []
        
        # Get colleges if field exists
        if hasattr(PYQ, 'college'):
            colleges = session.execute(text("SELECT DISTINCT college FROM pyqs WHERE college IS NOT NULL")).fetchall()
            filter_options['colleges'] = [c[0] for c in colleges] if colleges else []
        else:
            filter_options['colleges'] = []
        
        # Get courses if field exists
        if hasattr(PYQ, 'course'):
            courses = session.execute(text("SELECT DISTINCT course FROM pyqs WHERE course IS NOT NULL")).fetchall()
            filter_options['courses'] = [c[0] for c in courses] if courses else []
        else:
            filter_options['courses'] = []
        
        # Get branches if field exists
        if hasattr(PYQ, 'branch'):
            branches = session.execute(text("SELECT DISTINCT branch FROM pyqs WHERE branch IS NOT NULL")).fetchall()
            filter_options['branches'] = [b[0] for b in branches] if branches else []
        else:
            filter_options['branches'] = []
        
        # Get semester types if field exists
        if hasattr(PYQ, 'semester'):
            semesters = session.execute(text("SELECT DISTINCT semester FROM pyqs WHERE semester IS NOT NULL")).fetchall()
            filter_options['semesters'] = [s[0] for s in semesters] if semesters else []
        else:
            filter_options['semesters'] = []
        
        # Get units if field exists
        if hasattr(PYQ, 'unit'):
            units = session.execute(text("SELECT DISTINCT unit FROM pyqs WHERE unit IS NOT NULL")).fetchall()
            filter_options['units'] = [u[0] for u in units] if units else []
        else:
            filter_options['units'] = []
        
        return filter_options
        
    except Exception as e:
        print(f"Error getting filter options: {e}")
        return {
            'subjects': [],
            'colleges': [],
            'courses': [],
            'branches': [],
            'semesters': [],
            'units': []
        }


def validate_filters(filter_params: dict, session: Session) -> tuple:
    """
    Validate that the provided filters will return results from the database.
    Returns (is_valid: bool, message: str, result_count: int)
    """
    try:
        query = session.query(PYQ)
        
        for field, value in filter_params.items():
            if hasattr(PYQ, field):
                query = query.filter(getattr(PYQ, field) == value)
        
        count = query.count()
        
        if count == 0:
            return False, "No questions found matching the selected filters", 0
        else:
            return True, f"Found {count} questions matching the filters", count
            
    except Exception as e:
        return False, f"Error validating filters: {e}", 0


def get_filter_statistics(session: Session, filter_params: dict) -> dict:
    """
    Get statistics for the filtered dataset.
    """
    try:
        query = session.query(PYQ)
        
        for field, value in filter_params.items():
            if hasattr(PYQ, field):
                query = query.filter(getattr(PYQ, field) == value)
        
        pyqs = query.all()
        
        if not pyqs:
            return {"total": 0}
        
        stats = {
            "total": len(pyqs),
            "subjects": len(set([p.subject for p in pyqs if p.subject])),
            "subtopics": len(set([p.sub_topic for p in pyqs if p.sub_topic])),
            "years": len(set([p.year for p in pyqs if p.year])),
            "avg_marks": sum([p.marks for p in pyqs if p.marks]) / len([p for p in pyqs if p.marks]) if any(p.marks for p in pyqs) else 0
        }
        
        # Add frequency stats if available
        if hasattr(PYQ, 'frequency'):
            frequencies = [getattr(p, 'frequency', 1) for p in pyqs]
            stats["avg_frequency"] = sum(frequencies) / len(frequencies) if frequencies else 0
            stats["high_frequency_questions"] = len([f for f in frequencies if f > 2])
        
        return stats
        
    except Exception as e:
        print(f"Error getting filter statistics: {e}")
        return {"total": 0, "error": str(e)}