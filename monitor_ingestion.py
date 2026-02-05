#!/usr/bin/env python3
"""
Monitor ingestion progress by checking database counts periodically.
"""
import asyncio
import asyncpg
import os
import time
from dotenv import load_dotenv

load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

async def monitor():
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT')),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
    print("Monitoring ingestion progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    prev_docs = 0
    start_time = time.time()
    
    try:
        while True:
            docs = await conn.fetchval('SELECT COUNT(*) FROM documentos')
            chunks = await conn.fetchval('SELECT COUNT(*) FROM chunks')
            
            elapsed = time.time() - start_time
            docs_added = docs - prev_docs if prev_docs > 0 else 0
            rate = docs_added / 10 if docs_added > 0 else 0  # docs per second (10s interval)
            
            # Estimate: 1205 total files in Resolução folder
            target = 1205
            remaining = target - docs
            eta_seconds = remaining / rate if rate > 0 else 0
            
            print(f"[{time.strftime('%H:%M:%S')}] Docs: {docs}/{target} | Chunks: {chunks} | Rate: {rate:.2f} docs/s | ETA: {eta_seconds/60:.1f} min")
            
            prev_docs = docs
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(monitor())
