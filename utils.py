# utils.py

import fitz  # PyMuPDF
import re
from typing import List, Optional

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extracts text from each page of the PDF and returns
    a list of text chunks (one chunk per page).
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[str]: List of text content, one string per page
        
    Raises:
        Exception: If PDF cannot be opened or processed
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # Extract page text as plain text
            
            # Clean up the text (remove excessive whitespace)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Only add non-empty pages
            if text:
                pages_text.append(text)
            else:
                pages_text.append(f"[Page {page_num + 1} - No readable text found]")
                
        return pages_text
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF '{pdf_path}': {str(e)}")
    
    finally:
        # Ensure document is properly closed
        if doc:
            doc.close()

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Simple chunking logic - splits text by character count.
    Note: This function is not used in the main application but kept for utility.
    """
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def highlight_chunks(text: str) -> str:
    """
    HTML highlighting of text chunks for display.
    Note: This function is not used in the main application but kept for utility.
    """
    if not text:
        return ""
    
    return f"<div style='background-color:#f9f9f9;padding:10px;border-left:4px solid #4CAF50;margin-bottom:10px'>{text}</div>"

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,!?;:()\-"]', ' ', text)
    
    return text

def validate_pdf_file(pdf_path: str) -> bool:
    """
    Validate if the file is a readable PDF.
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count > 0
    except:
        return False
