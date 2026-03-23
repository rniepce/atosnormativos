
import os
import sys
import asyncio
import asyncpg
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.ingestion.common import extract_text, chunk_text, build_metadata_from_path, get_embeddings

# Load environment
load_dotenv(".env")
sys.path.insert(0, ".")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingestion_cpu.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client for embeddings
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', 'https://assistente-web-resource.cognitiveservices.azure.com/')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2025-01-01-preview')
if not AZURE_API_KEY:
    logger.error("AZURE_API_KEY not found in environment")
    sys.exit(1)

embedding_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
logger.info("Azure OpenAI embedding client initialized (text-embedding-3-large, 1024-dim)")

SOURCE_DIR = Path(os.getenv("SOURCE_DIR", str(Path(__file__).parent.parent.parent / "data")))

async def main():
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB")
        )
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    # Check existing files
    logger.info("Checking database for existing files...")
    existing = await conn.fetch("SELECT filename FROM documentos")
    existing_set = {r["filename"] for r in existing}
    logger.info(f"Found {len(existing_set)} already ingested files.")

    # Find files to process
    all_files = []
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in [".doc", ".docx"] and not f.name.startswith("~$"):
                    all_files.append(f)
    
    files_to_process = [f for f in all_files if f.name not in existing_set]
    logger.info(f"Files to process: {len(files_to_process)}")

    success, failed = 0, 0
    total_chunks = 0
    start_time = time.time()
    
    # Process files
    for i, file_path in enumerate(files_to_process, 1):
        try:
            # Stats every 10 files
            if i % 10 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = (success + failed) / elapsed if elapsed > 0 else 0
                eta = (len(files_to_process) - i) / rate if rate > 0 else 0
                logger.info(f"[{i}/{len(files_to_process)}] Success: {success} | Failed: {failed} | Rate: {rate:.2f} docs/s | ETA: {eta/3600:.1f}h")

            # 1. Extract
            logger.info(f"Extracting {file_path.name}...")
            text = extract_text(file_path, timeout=20)
            if not text or len(text.strip()) < 50:
                logger.info(f"Skipping {file_path.name} (empty or no text)")
                failed += 1
                continue
            
            # 2. Chunk
            meta = build_metadata_from_path(file_path)
            chunks = chunk_text(text)
            logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks")
            
            if not chunks:
                failed += 1
                continue

            # 3. Embed via Azure OpenAI
            try:
                logger.info(f"Embedding {file_path.name}...")
                embeddings = get_embeddings(chunks, embedding_client)
                logger.info(f"Embedded {file_path.name}!")
            except Exception as e:
                logger.error(f"Error embedding {file_path.name}: {e}")
                failed += 1
                continue

            # 4. Save
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags) 
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                    file_path.name, str(file_path), meta["tipo"], meta["numero"], meta["ano"], meta["orgao"], meta["status"], meta["assunto_resumo"], meta["tags"]
                )
                
                chunk_data = [(doc_id, chunk, str(list(emb))) for chunk, emb in zip(chunks, embeddings)]
                await conn.executemany(
                    "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                    chunk_data
                )
            
            success += 1
            total_chunks += len(chunks)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            failed += 1

    await conn.close()
    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"Total Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/3600:.1f}h")
    logger.info("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ingestion stopped by user.")
