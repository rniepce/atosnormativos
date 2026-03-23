import os
import re
import logging
import asyncpg

logger = logging.getLogger(__name__)

# Module-level connection pool (initialized via init_pool / close_pool)
_pool: asyncpg.Pool | None = None


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

    @property
    def safe_dsn(self):
        """Return DSN with password masked — safe for logging."""
        return re.sub(
            r"://([^:]+):([^@]+)@",
            r"://\1:****@",
            self.dsn,
        )


async def init_pool(min_size: int = 2, max_size: int = 10):
    """Create the module-level asyncpg connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    params = StorageParams()
    logger.info(f"Creating connection pool: {params.safe_dsn}")
    _pool = await asyncpg.create_pool(
        params.dsn,
        min_size=min_size,
        max_size=max_size,
    )
    return _pool


async def close_pool():
    """Close the module-level connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed")


async def get_db_connection():
    """Return a connection from the pool, or create a direct connection as fallback."""
    global _pool
    if _pool is not None:
        return await _pool.acquire()
    # Fallback for scripts that don't initialise the pool (ingestion scripts, etc.)
    params = StorageParams()
    try:
        conn = await asyncpg.connect(params.dsn)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DB: {params.safe_dsn} — {e}")
        raise e


async def release_db_connection(conn):
    """Release a connection back to the pool (or close if no pool)."""
    global _pool
    if _pool is not None:
        await _pool.release(conn)
    else:
        await conn.close()
