"""
Full ingestion script for 13k+ documents.
Uses folder name as document type (tipo).
"""
import sys
import os
import asyncio
import subprocess
import re
import time
import logging
from pathlib import Path
from typing import Set, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

from sentence_transformers import SentenceTransformer
import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/ingestion_full.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Source directory
SOURCE_DIR = Path('/Users/rafaelpimentel/Downloads/word')

# Initialize embedding model
print("Loading embedding model...", flush=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!", flush=True)


def extract_text(file_path: Path, timeout: int = 30) -> Optional[str]:
    """Extract text from .doc/.docx using macOS textutil."""
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', str(file_path)],
            capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout extracting {file_path}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting {file_path}: {e}")
        return None


def extract_metadata_from_path(file_path: Path) -> dict:
    """
    Extract metadata from file path.
    The parent folder name is the document type.
    """
    folder_name = file_path.parent.name
    filename = file_path.stem.lower()
    
    # Default metadata
    metadata = {
        'tipo': folder_name,
        'numero': '0',
        'ano': 0,
        'orgao': None,
        'status': 'VIGENTE',
        'assunto_resumo': f'{folder_name}',
        'tags': []
    }
    
    # Try to extract number and year from filename
    # Pattern: something + number + year (4 digits)
    # Examples: re00011961.doc, port_123_2024.doc, pc0012025.doc
    patterns = [
        r'(\d{3,5})(\d{4})$',  # re00011961 -> 1, 1961
        r'(\d{1,4})[_\-](\d{4})',  # port_123_2024
        r'(\d{1,4})(\d{4})$',  # General fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            metadata['numero'] = str(int(match.group(1)))
            year = int(match.group(2))
            if 1900 <= year <= 2030:
                metadata['ano'] = year
            break
    
    # Extract orgao from folder name if it contains institution codes
    orgao_patterns = {
        'TJMG': 'Tribunal de Justiça de MG',
        'CGJ': 'Corregedoria Geral de Justiça',
        'SEF': 'Secretaria de Fazenda',
        'OAB': 'Ordem dos Advogados do Brasil',
        'MPMG': 'Ministério Público de MG',
        'DPMG': 'Defensoria Pública de MG',
        'Presidência': 'Presidência do TJMG',
        'Vice-Presidência': 'Vice-Presidência do TJMG',
    }
    
    for key, value in orgao_patterns.items():
        if key.lower() in folder_name.lower():
            metadata['orgao'] = value
            break
    
    if not metadata['orgao']:
        metadata['orgao'] = 'TJMG'
    
    return metadata


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Simple text chunking with overlap."""
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunks.append(text[start:start+chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c] or [text[:chunk_size]] if text else []


async def get_db_connection():
    """Get database connection."""
    return await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT')),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )


async def get_existing_files(conn) -> Set[str]:
    """Get set of already ingested filenames."""
    records = await conn.fetch("SELECT filename FROM documentos")
    return {r['filename'] for r in records}


async def main():
    """Main ingestion loop."""
    try:
        conn = await get_db_connection()
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    # Get existing files to skip (for resume capability)
    existing_files = await get_existing_files(conn)
    logger.info(f"Found {len(existing_files)} already ingested files")

    # Collect all files from all subdirectories
    all_files = []
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in ['.doc', '.docx'] and not f.name.startswith('~$'):
                    all_files.append(f)
    
    # Filter out already processed
    files_to_process = [f for f in all_files if f.name not in existing_files]
    
    logger.info(f"Total files found: {len(all_files)}")
    logger.info(f"New files to process: {len(files_to_process)}")

    success, failed = 0, 0
    start_time = time.time()

    for i, file_path in enumerate(files_to_process, 1):
        # Progress logging every 50 files
        if i % 50 == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(files_to_process) - i) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i}/{len(files_to_process)} | "
                f"Success: {success} | Failed: {failed} | "
                f"Rate: {rate:.1f} docs/s | ETA: {eta/60:.1f} min"
            )

        # Extract text
        text = extract_text(file_path, timeout=20)
        if not text or len(text.strip()) < 50:
            logger.debug(f"Skipping {file_path.name}: too short or extraction failed")
            failed += 1
            continue

        # Get metadata from path
        metadata = extract_metadata_from_path(file_path)
        
        # Chunk text
        chunks = chunk_text(text)
        
        try:
            # Generate embeddings
            embeddings = model.encode(chunks, show_progress_bar=False)
            
            # Insert into database
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    '''INSERT INTO documentos 
                       (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags) 
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
                       RETURNING id''',
                    file_path.name,
                    str(file_path),
                    metadata['tipo'],
                    metadata['numero'],
                    metadata['ano'],
                    metadata['orgao'],
                    metadata['status'],
                    metadata['assunto_resumo'],
                    metadata['tags']
                )
                
                # Insert chunks
                chunk_data = [
                    (doc_id, chunk, str(list(emb))) 
                    for chunk, emb in zip(chunks, embeddings)
                ]
                await conn.executemany(
                    'INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)',
                    chunk_data
                )
            
            success += 1
            
        except Exception as e:
            logger.error(f"Error saving {file_path.name}: {e}")
            failed += 1

    await conn.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/60:.1f} min")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
