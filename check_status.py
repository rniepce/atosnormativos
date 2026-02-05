import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def check():
    load_dotenv()
    try:
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST'),
            port=int(os.getenv('POSTGRES_PORT')),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB')
        )
        count = await conn.fetchval("SELECT COUNT(*) FROM documentos")
        print(f"Total documents in DB: {count}")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check())
