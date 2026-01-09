## 🎯IntelliJect


![Untitled video - Made with Clipchamp (5)](https://github.com/user-attachments/assets/5e9c7030-172d-4a20-884f-349f1d6abe8f)


AI-powered exam-oriented study assistant using RAG, embeddings, and semantic search

## IntelliJect is an AI-powered study assistant that enriches student notes with contextually relevant Previous Year Questions (PYQs). It helps students study in an exam-oriented manner by automatically mapping their notes to syllabus-aligned PYQs using embeddings, semantic search, and a Retrieval-Augmented Generation (RAG) pipeline.


## Key Features

  1. Notes Matcher (RAG-based)
    Upload PDF notes
    Automatically extract and chunk content
    Semantically match each chunk with relevant PYQs
    Highlight exact answer lines directly from notes
    Display matched PYQs alongside notes for contextual learning     
     ![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/7041b618-94f8-47ce-bc0f-87596e99cbfa)
     
  2.Smart Database Builder
    Upload scanned or digital question papers
    OCR-based text extraction with preprocessing
    LLM-powered parsing to structure questions with metadata
    Deduplication using question hashing
    Manual review and edit before database insertion   
    ![db_comp](https://github.com/user-attachments/assets/23283899-abbb-43a4-9905-cba62062e32e)
    
  3. Contextual PYQ Search Engine
    Filter PYQs by subject, college, branch, semester, unit, marks, frequency
    Semantic search using vector embeddings (FAISS)
    Difficulty-based and frequency-based question recommendations 
    ![search](https://github.com/user-attachments/assets/5f4bb5cc-29e5-477b-b3d8-4821bfc875c6)

  4. AI Study Planner
    Generates a day-wise study plan based on PYQ frequency
    Prioritizes high-weightage topics
    Estimates daily workload and time 

 ![study planner](https://github.com/user-attachments/assets/c2f0b271-f10c-4eaa-9f89-0e6749bd3dbc)

## System Architecture (High-Level)

    Frontend: Streamlit (multi-tab interactive UI)
    Backend Logic:
    Python
    LangChain for document abstraction
    OpenAI embeddings for vector representation
    FAISS for fast semantic retrieval
    Database:
    PostgreSQL hosted on AWS RDS
    Structured schema for PYQs, syllabus, and metadata
    Authentication:
    Supabase (email/password auth)
    Deployment:
    AWS EC2 for application hosting
    AWS RDS for persistent storage
    
  <img width="874" height="478" alt="image" src="https://github.com/user-attachments/assets/9f1f86d4-9a0b-4e73-9b31-0a3c23674541" />


## Tech Stack

    Layer : Technologies
    Frontend : Streamlit
    Backend : Python, LangChain
    LLM & Embeddings : OpenAI
    Vector Search : FAISS
    OCR : PyMuPDF, OpenCV, Tesseract
    Database : PostgreSQL (AWS RDS)
    Auth : Supabase
    Cloud : AWS EC2
    
    Language: Python
    Frontend: Streamlit
    LLM & Embeddings: OpenAI (via LangChain)
    Vector Store: FAISS
    OCR: PyMuPDF, OpenCV, Tesseract (pytesseract)
    Database: PostgreSQL (SQLAlchemy ORM)
    Authentication: Supabase
    Cloud: AWS EC2, AWS RDS

## How IntelliJect Works (Notes Matcher Flow)

        A[Upload Notes PDF] --> B[Text Extraction / OCR]
        B --> C[Sentence Chunking]
        C --> D[OpenAI Embeddings]
        D --> E[FAISS Semantic Search]
        E --> F[Relevant PYQs]
        F --> G[Answer Extraction via LLM]
        G --> H[Highlighted Answers in UI]
        
    1.User uploads a PDF of notes
    2.Text is extracted (OCR fallback for scanned PDFs)
    3.Notes are chunked into semantically meaningful units
    4.Each chunk is embedded into a vector
    5.PYQs are filtered using syllabus metadata
    6.FAISS performs semantic search to retrieve top-matching PYQs
    7.An LLM extracts exact answer sentences from the notes
    8.Results are displayed with highlighted answers
    
## Database Design (Core Table: PYQ)
    Each PYQ record contains: - Question text - Subject, unit, sub-topic - Branch, college, course - Year, semester, marks - Frequency (number of times asked) - Hash for        deduplication - Verification status
    Indexes are added on frequently queried fields (subject, unit, branch, frequency) for performance.

## Security & Configuration

    >> All secrets managed via environment variables:
    OPENAI_API_KEY
    DATABASE_URL
    SUPABASE_URL
    SUPABASE_ANON_KEY
    TESSERACT_CMD
    >> No credentials are hard-coded
    >> Database access via SQLAlchemy ORM (prevents SQL injection)
    >> Designed for deployment behind HTTPS in production

## Deployment Overview


    >> PostgreSQL database deployed on AWS RDS
    >> Application hosted on AWS EC2
    >> Environment variables configured on server
    >> Database populated via JSON migration and Smart DB Builder
    >> Streamlit app served to users
    
  ## Scalability Considerations
  
    >> Connection pooling for database access
    >> Metadata filtering before semantic search to reduce vector load
    >> Future-ready for:
        Persistent FAISS indices
        Caching frequent queries
        Horizontal scaling with containers

## Use Cases

    >> Students preparing for university exams
    >> Exam-oriented revision using PYQs
    >> Identifying high-frequency and high-weightage topics
    >> Structured study planning

## Future Enhancements

    >> Handwritten notes OCR support
    >> Weak-topic detection based on study patterns
    >> Multi-domain expansion (medical, CA, banking exams)
    >> Analytics dashboard for student performance

## Contributors
    Both– Backend architecture, RAG pipeline, embeddings, FAISS integration, database design
    Aqsa Laraib – AWS deployment, Frontend 
    Dishaa Mehra – Database design

## License

    This project is currently for academic and research purposes. Licensing can be updated as the project evolves.

IntelliJect — Study smarter. Study exam-oriented.


