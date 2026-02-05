import fitz  # PyMuPDF
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts text from a PDF file using PyMuPDF.
    Returns the full text of the document.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}. Document might be scanned image.")
            return None
            
        return text
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None

def is_searchable_pdf(pdf_path: str) -> bool:
    """
    Simple check to see if PDF has text layer.
    """
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.get_text().strip():
                return True
        return False
    except Exception:
        return False

def extract_text_from_doc_docx(file_path: str) -> Optional[str]:
    """
    Extracts text from .doc and .docx files using macOS 'textutil'.
    Returns the full text of the document.
    """
    import subprocess
    import shutil
    
    if not shutil.which("textutil"):
        logger.error("textutil not found. This function requires macOS.")
        return None

    try:
        # textutil -convert txt -stdout "file_path"
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
        return None

