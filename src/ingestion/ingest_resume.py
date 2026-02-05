"""
Resumable ingestion script - skips already processed files.
Uses sentence-transformers for embeddings and filename-based classification.
Has improved connection handling with reconnection logic.
"""
import asyncio
import os
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sentence_transformers import SentenceTransformer
import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize local embedding model
EMBEDDING_MODEL = None

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model...", flush=True)
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!", flush=True)
    return EMBEDDING_MODEL

def extract_text_from_doc_docx(file_path: str) -> Optional[str]:
    """Extract text from .doc/.docx using macOS textutil."""
    import subprocess
    import shutil
    
    if not shutil.which("textutil"):
        logger.error("textutil not found. This function requires macOS.")
        return None

    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", file_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout extracting text from {file_path}")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
        return None

def classify_from_filename(filename: str, parent_dir: str) -> Dict[str, Any]:
    """Extract metadata from filename and directory."""
    name_lower = filename.lower()
    
    metadata = {
        "tipo": "Desconhecido",
        "numero": "0",
        "ano": 0,
        "status": "VIGENTE",
        "assunto_resumo": f"Extraído de {parent_dir}",
        "tags": []
    }
    
    # Pattern: re00011961.doc -> Resolução 1/1961
    match = re.match(r'^(re|res)(\d{4})(\d{4})', name_lower)
    if match:
        metadata["tipo"] = "Resolução"
        metadata["numero"] = str(int(match.group(2)))
        metadata["ano"] = int(match.group(3))
        return metadata
    
    # Pattern: port123_2024 or similar
    match = re.match(r'^(port|portaria)[\D]*(\d+)[\D]*(\d{4})', name_lower)
    if match:
        metadata["tipo"] = "Portaria"
        metadata["numero"] = match.group(2)
        metadata["ano"] = int(match.group(3))
        return metadata
    
    # Pattern: prov123_2024 or similar
    match = re.match(r'^(prov|provimento)[\D]*(\d+)[\D]*(\d{4})', name_lower)
    if match:
        metadata["tipo"] = "Provimento"
        metadata["numero"] = match.group(2)
        metadata["ano"] = int(match.group(3))
        return metadata
    
    # Use parent directory as type hint
    parent_lower = parent_dir.lower()
    if "resolucao" in parent_lower or "resolução" in parent_lower:
        metadata["tipo"] = "Resolução"
    elif "portaria" in parent_lower:
        if "conjunta" in parent_lower:
            metadata["tipo"] = "Portaria Conjunta"
        else:
            metadata["tipo"] = "Portaria"
    elif "provimento" in parent_lower:
        metadata["tipo"] = "Provimento"
    elif "aviso" in parent_lower:
        metadata["tipo"] = "Aviso"
    elif "instrucao" in parent_lower or "instrução" in parent_lower:
        metadata["tipo"] = "Instrução"
    elif "emenda" in parent_lower:
        metadata["tipo"] = "Emenda Regimental"
    elif "ordem" in parent_lower:
        metadata["tipo"] = "Ordem de Serviço"
    
    # Try to extract any number and year from filename
    numbers = re.findall(r'\d+', filename)
    if len(numbers) >= 2:
        for num in reversed(numbers):
            if len(num) == 4 and 1900 <= int(num) <= 2100:
                metadata["ano"] = int(num)
                break
        metadata["numero"] = numbers[0].lstrip('0') or '0'
    
    return metadata

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple text chunking with overlap."""
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks if chunks else [text[:chunk_size]] if text else []

async def get_db_connection():
    """Get database connection using environment variables."""
    return await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )

async def get_processed_filenames(conn) -> Set[str]:
    """Get set of already processed filenames from database."""
    rows = await conn.fetch("SELECT filename FROM documentos")
    return {row['filename'] for row in rows}

async def process_file(file_path: Path, model: SentenceTransformer, conn) -> bool:
    """Process a single file: extract text, classify, chunk, embed, store."""
    filename = file_path.name
    logger.info(f"Processing: {filename}")
    
    # 1. Extract Text
    text = extract_text_from_doc_docx(str(file_path))
    if not text or len(text.strip()) < 50:
        logger.warning(f"Skipping {filename}: No text extracted or too short.")
        return False
    
    # 2. Classify from filename
    metadata = classify_from_filename(filename, file_path.parent.name)
    logger.info(f"  Classified as: {metadata['tipo']} {metadata['numero']}/{metadata['ano']}")
    
    # 3. Chunk the text (limit to prevent memory issues)
    chunks = chunk_text(text)
    if len(chunks) > 500:
        logger.warning(f"  Document has {len(chunks)} chunks, limiting to 500")
        chunks = chunks[:500]
    logger.info(f"  Created {len(chunks)} chunks")
    
    # 4. Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    # 5. Store in database
    try:
        async with conn.transaction():
            # Insert document
            doc_id = await conn.fetchval("""
                INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, status_vigencia, assunto_resumo, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, 
                filename,
                str(file_path),
                metadata.get("tipo"),
                metadata.get("numero"),
                metadata.get("ano"),
                metadata.get("status"),
                metadata.get("assunto_resumo"),
                metadata.get("tags", [])
            )
            
            # Insert chunks with embeddings
            chunk_data = [
                (doc_id, chunk, str(list(emb)))
                for chunk, emb in zip(chunks, embeddings)
            ]
            
            await conn.executemany(
                "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                chunk_data
            )
            
        logger.info(f"  ✓ Saved document ID {doc_id} with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Database error: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Resumable ingestion - skips already processed files")
    parser.add_argument("--dir", required=True, help="Directory to ingest")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 for all)")
    parser.add_argument("--batch-size", type=int, default=100, help="Reconnect to DB every N files")
    args = parser.parse_args()
    
    root_dir = Path(args.dir)
    if not root_dir.exists():
        print(f"Directory not found: {root_dir}")
        return

    # Load embedding model
    model = get_embedding_model()
    
    # Get database connection
    try:
        conn = await get_db_connection()
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    # Get already processed filenames
    logger.info("Fetching list of already processed files...")
    processed_filenames = await get_processed_filenames(conn)
    logger.info(f"Found {len(processed_filenames)} already processed files in database")
    
    # Find all files
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".doc", ".docx")) and not file.startswith("~$"):
                all_files.append(Path(root) / file)
    
    # Filter out already processed files
    files_to_process = [f for f in all_files if f.name not in processed_filenames]
    
    logger.info(f"Total files found: {len(all_files)}")
    logger.info(f"Files to process (new): {len(files_to_process)}")
    
    if args.limit > 0:
        files_to_process = files_to_process[:args.limit]
        logger.info(f"Limited to: {len(files_to_process)} files")
    
    if not files_to_process:
        logger.info("No new files to process!")
        await conn.close()
        return
    
    # Process files with periodic reconnection
    success = 0
    failed = 0
    batch_size = args.batch_size
    
    for i, file_path in enumerate(files_to_process, 1):
        # Reconnect every batch_size files to avoid connection timeout
        if i > 1 and (i - 1) % batch_size == 0:
            logger.info(f"Reconnecting to database (every {batch_size} files)...")
            try:
                await conn.close()
            except:
                pass
            try:
                conn = await get_db_connection()
                logger.info("Reconnected successfully")
            except Exception as e:
                logger.error(f"Failed to reconnect: {e}")
                break
        
        logger.info(f"[{i}/{len(files_to_process)}] Processing {file_path.name}")
        try:
            if await process_file(file_path, model, conn):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed += 1
            # Try to reconnect on error
            try:
                await conn.close()
            except:
                pass
            try:
                conn = await get_db_connection()
                logger.info("Reconnected after error")
            except Exception as e2:
                logger.error(f"Failed to reconnect after error: {e2}")
                break
    
    try:
        await conn.close()
    except:
        pass
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingestion complete!")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
