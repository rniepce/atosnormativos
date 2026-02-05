-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For keyword search optimization

-- Tabela de Documentos (Metadados Pai)
CREATE TABLE IF NOT EXISTS documentos (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    gcs_uri TEXT,
    tipo VARCHAR(50),
    numero VARCHAR(20),
    ano INTEGER,
    orgao VARCHAR(100), -- Órgão emissor (Presidência, Corregedoria, etc.)
    status_vigencia VARCHAR(20), -- 'VIGENTE', 'REVOGADO'
    assunto_resumo TEXT,
    tags TEXT[], 
    data_upload TIMESTAMP DEFAULT NOW()
);

-- Tabela de Vetores (Chunks)
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    documento_id INTEGER REFERENCES documentos(id) ON DELETE CASCADE,
    conteudo_texto TEXT NOT NULL, -- O trecho da lei
    embedding vector(384) -- O vetor gerado pelo all-MiniLM-L6-v2 (384-dim)
);

-- Índices para performance
-- HNSW Index for fast approximate nearest neighbor search
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- Indexes for filtering and sorting
CREATE INDEX ON documentos (status_vigencia);
CREATE INDEX ON documentos (ano);
CREATE INDEX ON documentos (tipo);
CREATE INDEX ON documentos USING GIN (tags);

-- Full text search index (optional, useful for hybrid search keyword matching)
CREATE INDEX ON chunks USING GIN (to_tsvector('portuguese', conteudo_texto));
