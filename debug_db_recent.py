
import asyncio
import os
from dotenv import load_dotenv
import asyncpg
from pathlib import Path

# Load env from the correct path
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

async def main():
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB")
        )
        print("Connected to DB successfully.")

        # 1. Count by year
        print("\n--- Documents by Year ---")
        rows = await conn.fetch("SELECT ano, COUNT(*) as qtd FROM documentos GROUP BY ano ORDER BY ano DESC")
        for row in rows:
            print(f"Year {row['ano']}: {row['qtd']} docs")

        # 2. Top 20 most recent files
        print("\n--- Recent Files ---")
        rows = await conn.fetch("SELECT filename, ano, created_at FROM documentos ORDER BY ano DESC, created_at DESC LIMIT 20")
        for row in rows:
            print(f"{row['filename']} (Year: {row['ano']})")

        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
