import os
import logging
import asyncpg

logger = logging.getLogger(__name__)

class StorageParams:
    def __init__(self):
        self.user = os.getenv("POSTGRES_USER", "admin")
        self.password = os.getenv("POSTGRES_PASSWORD", "password")
        self.database = os.getenv("POSTGRES_DB", "tjmg_rag")
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")

    @property
    def dsn(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

async def get_db_connection():
    params = StorageParams()
    try:
        conn = await asyncpg.connect(params.dsn)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        raise e
