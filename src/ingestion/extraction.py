import fitz  # PyMuPDF
import logging
import platform
import subprocess
import shutil
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
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text() for page in doc)

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
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if page.get_text().strip():
                    return True
            return False
    except Exception:
        return False


def extract_text_from_doc_docx(file_path: str) -> Optional[str]:
    """
    Extracts text from .doc and .docx files.
    Uses macOS 'textutil' when available; falls back to python-docx for .docx
    and subprocess antiword for .doc on Linux.
    """
    # Try textutil first (macOS)
    if shutil.which("textutil"):
        try:
            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", file_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting {file_path} with textutil: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            return None

    # Fallback for Linux / environments without textutil
    lower_path = file_path.lower()

    if lower_path.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
            return text if text.strip() else None
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"Error extracting .docx {file_path}: {e}")
            return None

    if lower_path.endswith(".doc"):
        # Try antiword first
        if shutil.which("antiword"):
            try:
                result = subprocess.run(
                    ["antiword", file_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout
            except Exception as e:
                logger.error(f"antiword failed for {file_path}: {e}")
                return None
        else:
            logger.error(f"Cannot extract .doc on this platform. Install 'antiword' or run on macOS.")
            return None

    logger.error(f"Unsupported file format: {file_path}")
    return None

