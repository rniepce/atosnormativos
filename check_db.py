import sys
import asyncio
import asyncpg

async def main():
    dsn = "postgresql://postgres:pvjPoRKOeQOVmZNCulAVYXqpWmnefbsa@trolley.proxy.rlwy.net:21494/railway"
    print("Connecting to Railway database...")
    try:
        conn = await asyncpg.connect(dsn)
        print("✅ Successfully connected!")
        
        # 1. Check total documents
        total_docs = await conn.fetchval("SELECT COUNT(*) FROM documentos")
        print(f"\nTotal documents in DB: {total_docs}")
        
        if total_docs > 0:
            # 2. Check document types and sample metadata
            print("\nDocument Types distribution:")
            types = await conn.fetch("SELECT tipo, COUNT(*) as count FROM documentos GROUP BY tipo ORDER BY count DESC")
            for t in types:
                print(f"  - {t['tipo']}: {t['count']}")
                
            print("\nSample Documents (Latest 5):")
            recent = await conn.fetch("SELECT id, filename, tipo, status_vigencia FROM documentos ORDER BY id DESC LIMIT 5")
            for r in recent:
                print(f"  - ID: {r['id']}, File: {r['filename']}, Type: {r['tipo']}, Status: {r['status_vigencia']}")
                
        # 3. Check chunks and vector dimensions
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        print(f"\nTotal chunks in DB: {total_chunks}")
        
        if total_chunks > 0:
            # Try to get the vector dimension
            try:
                # pgvector specific syntax to get dimensions
                dim = await conn.fetchval("SELECT vector_dims(embedding) FROM chunks LIMIT 1")
                print(f"Vector Dimensions: {dim}")
            except Exception as e:
                print(f"Could not determine vector dimension: {e}")
                
            print("\nSample Chunks (Latest 2):")
            recent_chunks = await conn.fetch("SELECT id, documento_id, length(conteudo_texto) as text_len FROM chunks ORDER BY id DESC LIMIT 2")
            for c in recent_chunks:
                print(f"  - Chunk ID: {c['id']}, Doc ID: {c['documento_id']}, Text Length: {c['text_len']} chars")

        await conn.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
