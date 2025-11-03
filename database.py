from sqlalchemy import create_engine, Column, Integer, String, DateTime, Index, Text, Float, Boolean, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
import datetime
import os
import hashlib
import re
from sqlalchemy import text

load_dotenv()

# Create the full database URL from environment variable or fallback
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables")
try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=True  # Add this for debugging SQL statements
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    print("✅ Database engine created successfully")
except Exception as e:
    print(f"❌ Error creating database engine: {e}")
    raise

# Declare Base
Base = declarative_base()

# Enhanced PYQ table model with Smart Database Builder fields
class PYQ(Base):
    __tablename__ = "pyqs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    question_hash = Column(String(255), index=True)  # For duplicate detection
    subject = Column(String(255), nullable=False, index=True)
    sub_topic = Column(String(255), index=True)
    unit = Column(String(255), index=True)  # Unit 1, Unit 2, etc.
    year = Column(String(255), index=True)  # Can store multiple years: "2023,2024"
    semester = Column(String(255), index=True)  # mid-sem, end-sem, quiz, assignment
    branch = Column(String(255), index=True)  # CSE-AI, CSE, ECE, etc.
    marks = Column(Float)  # Supports decimal marks like 2.5
    frequency = Column(Integer, default=1, index=True)  # How many times this question appeared
    college = Column(String(255))  # College/University name
    course = Column(String(255))  # B.Tech, M.Tech, etc.
    verified = Column(Boolean, default=False)  # Manual verification flag
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    __table_args__ = (
        # Add unique constraint on question_hash to prevent duplicates at DB level
        UniqueConstraint('question_hash', name='unique_question_hash'),
        # Additional indexes for performance
        Index('idx_pyqs_subject_unit', 'subject', 'unit'),
        Index('idx_pyqs_branch_year', 'branch', 'year'),
        Index('idx_pyqs_frequency_desc', 'frequency'),
    )

    def __repr__(self):
        return (
            f"<PYQ(id={self.id}, subject='{self.subject}', "
            f"sub_topic='{self.sub_topic}', branch='{self.branch}', "
            f"frequency={self.frequency}, marks={self.marks})>"
        )

    @classmethod
    def generate_hash(cls, question_text: str) -> str:
        """Generate a consistent hash for question text"""
        if not question_text:
            return ""
        
        # Normalize the question text
        normalized = re.sub(r'\s+', ' ', question_text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Generate MD5 hash
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def set_question_hash(self):
        """Set the question hash based on question text"""
        self.question_hash = self.generate_hash(self.question)

# New table for syllabus context (helps with better question extraction)
class Syllabus(Base):
    __tablename__ = "syllabus"
    
    id = Column(Integer, primary_key=True, index=True)
    college = Column(String(255), nullable=False)
    course = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False, index=True)
    unit = Column(String(255), nullable=False, index=True)
    topics = Column(Text)  # JSON string of topics and subtopics
    description = Column(Text)  # Detailed unit description
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    __table_args__ = (
        Index('idx_syllabus_subject_unit', 'subject', 'unit'),
        UniqueConstraint('college', 'course', 'subject', 'unit', name='unique_syllabus_entry'),
    )

    def __repr__(self):
        return f"<Syllabus(subject='{self.subject}', unit='{self.unit}', college='{self.college}')>"

# File tracking table (optional - for audit trail)
class FileRecord(Base):
    __tablename__ = "file_records"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(100), nullable=False)  # question_paper, syllabus
    college = Column(String(255), nullable=False)
    course = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False, index=True)
    questions_extracted = Column(Integer, default=0)
    processing_status = Column(String(100), default="pending", index=True)  # pending, completed, failed
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    processed_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_files_subject_status', 'subject', 'processing_status'),
        Index('idx_files_uploaded_at', 'uploaded_at'),
    )

    def __repr__(self):
        return f"<FileRecord(filename='{self.filename}', status='{self.processing_status}')>"



# Function to create all tables
def create_tables():
    """Create all database tables"""
    try:
        print("🔄 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Enhanced database tables created successfully!")
        print("Tables created:")
        print("- pyqs (enhanced with Smart Database Builder fields)")
        print("- syllabus (new table for syllabus context)")
        print("- file_records (new table for file tracking)")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False
# Function to test database connection
def test_connection():
    """Test database connection"""
    try:
        print("🔄 Testing database connection...")
        with SessionLocal() as session:
            result = session.execute(text("SELECT 1"))
            result.fetchone()
        print("✅ Database connection successful")
        return True, "Connection successful"
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False, f"Connection failed: {e}"
# Function to check if tables exist
def check_tables_exist():
    """Check if tables exist in the database"""
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        expected_tables = ['pyqs', 'syllabus', 'file_records']
        missing_tables = [table for table in expected_tables if table not in existing_tables]
        
        print(f"📋 Existing tables: {existing_tables}")
        
        if missing_tables:
            print(f"⚠️ Missing tables: {missing_tables}")
            return False, missing_tables
        else:
            print("✅ All required tables exist")
            return True, []
            
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return False, [str(e)]

# Function to get database session
def get_db():
    """Get database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions
def get_session():
    """Get a database session (for direct use)"""
    return SessionLocal()

def get_database_info():
    """Get database information and statistics"""
    try:
        with SessionLocal() as session:
            
            # Get table counts
            try:
                pyq_count = session.query(PYQ).count()
            except:
                pyq_count = 0
            
            try:
                syllabus_count = session.query(Syllabus).count()
            except:
                syllabus_count = 0
            
            try:
                file_count = session.query(FileRecord).count()
            except:
                file_count = 0
            
            # Get subject distribution
            try:
                subjects = session.execute(text(
                    "SELECT subject, COUNT(*) as count FROM pyqs GROUP BY subject ORDER BY count DESC"
                )).fetchall()
            except:
                subjects = []
            
            return {
                'pyq_count': pyq_count,
                'syllabus_count': syllabus_count,
                'file_count': file_count,
                'subjects': subjects,
                'connection_status': 'Connected'
            }
            
    except Exception as e:
        return {
            'error': str(e),
            'connection_status': 'Failed'
        }


# Initialize and test on import
if __name__ == "__main__":
    print("🚀 Starting database initialization...")
    
    # Test connection first
    success, message = test_connection()
    if not success:
        print("❌ Cannot proceed without database connection")
        exit(1)
    
    # Check if tables exist
    tables_exist, missing = check_tables_exist()
    
    if not tables_exist:
        print(f"📦 Creating missing tables: {missing}")
        if create_tables():
            print("✅ Database setup completed successfully!")
        else:
            print("❌ Database setup failed!")
            exit(1)
    else:
        print("✅ All tables already exist!")
    
    # Show database info
    info = get_database_info()
    print(f"📊 Database info: {info}")
