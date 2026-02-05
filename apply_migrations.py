"""
Script to apply database migrations for the new schema.
Adds 'orgao' column and recreates chunks table with 384-dim vectors.
"""
import asyncio
import os
from dotenv import load_dotenv
import asyncpg

load_dotenv()

async def run_migrations():
    """Apply database migrations."""
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )
    
    print("Connected to database. Applying migrations...")
    
    try:
        # Add orgao column if not exists
        print("1. Adding 'orgao' column to documentos...")
        await conn.execute("""
            ALTER TABLE documentos ADD COLUMN IF NOT EXISTS orgao VARCHAR(100)
        """)
        print("   ✓ Column added/exists")
        
        # Drop and recreate chunks table with correct vector dimension
        print("2. Recreating chunks table with 384-dim vectors...")
        await conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        await conn.execute("""
            CREATE TABLE chunks (
                id SERIAL PRIMARY KEY,
                documento_id INTEGER REFERENCES documentos(id) ON DELETE CASCADE,
                conteudo_texto TEXT NOT NULL,
                embedding vector(384)
            )
        """)
        print("   ✓ Chunks table recreated")
        
        # Create indexes
        print("3. Creating indexes...")
        await conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")
        await conn.execute("CREATE INDEX ON chunks USING GIN (to_tsvector('portuguese', conteudo_texto))")
        print("   ✓ Indexes created")
        
        # Clean documentos table
        print("4. Cleaning documentos table...")
        await conn.execute("DELETE FROM documentos")
        print("   ✓ Documentos cleaned")
        
        print("\n✅ Migrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(run_migrations())
