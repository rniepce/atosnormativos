import asyncio
from src.utils.db import get_db_connection

async def count_docs():
    conn = await get_db_connection()
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM documentos")
        print(f"Document count: {count}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(count_docs())
