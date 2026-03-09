"""
Comprehensive search quality test for the re-embedded database.
Tests both vector similarity (self-similarity) and semantic search relevance.
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import asyncpg
from openai import AzureOpenAI

# Azure config
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)


def embed_query(text: str):
    response = client.embeddings.create(
        input=[text], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM
    )
    return response.data[0].embedding


async def main():
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
    )

    print("=" * 70)
    print("🔍 TESTE DE QUALIDADE DO BANCO VETORIZADO")
    print("=" * 70)

    # ── TEST 1: Self-similarity ─────────────────────────────────
    print("\n" + "─" * 50)
    print("1️⃣  AUTO-SIMILARIDADE (chunk vs seus vizinhos)")
    print("─" * 50)

    sample = await conn.fetchrow("""
        SELECT c.id, c.conteudo_texto, d.tipo, d.numero, d.ano, d.filename, d.orgao
        FROM chunks c JOIN documentos d ON c.documento_id = d.id
        WHERE c.embedding IS NOT NULL AND LENGTH(c.conteudo_texto) > 300
        ORDER BY RANDOM() LIMIT 1
    """)

    if sample:
        print(f"\n  📄 Chunk de referência:")
        print(f"     {sample['tipo']} {sample['numero']}/{sample['ano']} ({sample['filename']})")
        print(f"     Órgão: {sample['orgao']}")
        print(f"     Texto: {sample['conteudo_texto'][:200]}...")

        neighbors = await conn.fetch("""
            SELECT c.id, d.filename, d.tipo, d.numero, d.ano, d.orgao,
                   1 - (c.embedding <=> (SELECT embedding FROM chunks WHERE id = $1)) as similarity
            FROM chunks c JOIN documentos d ON c.documento_id = d.id
            WHERE c.id != $1 AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> (SELECT embedding FROM chunks WHERE id = $1) ASC
            LIMIT 5
        """, sample['id'])

        print(f"\n  Top 5 vizinhos:")
        for i, n in enumerate(neighbors, 1):
            print(f"    {i}. Sim={float(n['similarity']):.4f} | {n['tipo']} {n['numero']}/{n['ano']} ({n['filename']}) | {n['orgao']}")

    # ── TEST 2: Semantic search with real queries ───────────────
    print("\n" + "─" * 50)
    print("2️⃣  BUSCA SEMÂNTICA (queries reais em português)")
    print("─" * 50)

    queries = [
        "férias de magistrado",
        "teletrabalho home office trabalho remoto",
        "corregedoria procedimento disciplinar",
        "organização judiciária comarcas varas",
        "plantão judiciário",
        "conciliação mediação resolução de conflitos",
        "regimento interno tribunal pleno",
        "licitação contrato administrativo",
        "gestão de pessoas recursos humanos",
        "inteligência artificial tecnologia",
    ]

    for query in queries:
        print(f"\n  🔎 Query: \"{query}\"")
        
        # Embed query with same model
        query_emb = embed_query(query)
        emb_str = str(query_emb)

        results = await conn.fetch("""
            SELECT c.id, d.filename, d.tipo, d.numero, d.ano, d.orgao,
                   d.assunto_resumo,
                   1 - (c.embedding <=> $1::vector) as similarity,
                   LEFT(c.conteudo_texto, 150) as trecho
            FROM chunks c JOIN documentos d ON c.documento_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> $1::vector ASC
            LIMIT 5
        """, emb_str)

        for i, r in enumerate(results, 1):
            sim = float(r['similarity'])
            emoji = "🟢" if sim > 0.5 else ("🟡" if sim > 0.3 else "🔴")
            print(f"    {i}. {emoji} Sim={sim:.4f} | {r['tipo']} {r['numero']}/{r['ano']} | {r['orgao']}")
            print(f"       Assunto: {(r['assunto_resumo'] or '')[:100]}")
            print(f"       Trecho: {r['trecho'][:120]}...")

    # ── TEST 3: Keyword-specific search ─────────────────────────
    print("\n" + "─" * 50)
    print("3️⃣  BUSCA POR CONCEITO ESPECÍFICO")
    print("─" * 50)

    specific_queries = [
        ("Portaria sobre teletrabalho", "Portaria"),
        ("Resolução sobre concurso público", "Resolução"),
        ("Provimento da Corregedoria", "Provimento Conjunto"),
    ]

    for query, expected_type in specific_queries:
        print(f"\n  🔎 Query: \"{query}\" (esperando tipo ~{expected_type})")
        
        query_emb = embed_query(query)
        emb_str = str(query_emb)

        results = await conn.fetch("""
            SELECT d.tipo, d.numero, d.ano, d.orgao,
                   1 - (c.embedding <=> $1::vector) as similarity,
                   d.assunto_resumo
            FROM chunks c JOIN documentos d ON c.documento_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> $1::vector ASC
            LIMIT 5
        """, emb_str)

        type_match = sum(1 for r in results if expected_type.lower() in (r['tipo'] or '').lower())
        print(f"    Match de tipo: {type_match}/5 resultados são '{expected_type}'")
        for i, r in enumerate(results, 1):
            sim = float(r['similarity'])
            match = "✅" if expected_type.lower() in (r['tipo'] or '').lower() else "  "
            print(f"    {i}. {match} Sim={sim:.4f} | {r['tipo']} {r['numero']}/{r['ano']} | {(r['assunto_resumo'] or '')[:80]}")

    # ── TEST 4: Similarity distribution ─────────────────────────
    print("\n" + "─" * 50)
    print("4️⃣  DISTRIBUIÇÃO DE SIMILARIDADE")
    print("─" * 50)

    test_query = "organização judiciária do tribunal de justiça"
    query_emb = embed_query(test_query)
    emb_str = str(query_emb)

    dist = await conn.fetch("""
        SELECT 
            CASE 
                WHEN 1 - (embedding <=> $1::vector) >= 0.6 THEN '0.6+'
                WHEN 1 - (embedding <=> $1::vector) >= 0.5 THEN '0.5-0.6'
                WHEN 1 - (embedding <=> $1::vector) >= 0.4 THEN '0.4-0.5'
                WHEN 1 - (embedding <=> $1::vector) >= 0.3 THEN '0.3-0.4'
                WHEN 1 - (embedding <=> $1::vector) >= 0.2 THEN '0.2-0.3'
                ELSE '<0.2'
            END as faixa,
            COUNT(*) as qtd
        FROM chunks
        WHERE embedding IS NOT NULL
        GROUP BY faixa
        ORDER BY faixa DESC
    """, emb_str)

    print(f"  Query: \"{test_query}\"")
    for row in dist:
        bar = "█" * min(int(row['qtd'] / 100), 50)
        print(f"    {row['faixa']:>7}: {row['qtd']:>6} chunks  {bar}")

    # ── TEST 5: Cross-language sanity check ─────────────────────
    print("\n" + "─" * 50)
    print("5️⃣  TESTE MULTILÍNGUE (PT vs EN)")
    print("─" * 50)

    pt_query = "férias de magistrado e juiz"
    en_query = "judge vacation and leave"

    pt_emb = embed_query(pt_query)
    en_emb = embed_query(en_query)

    pt_results = await conn.fetch("""
        SELECT d.filename, d.tipo, d.numero, d.ano,
               1 - (c.embedding <=> $1::vector) as similarity
        FROM chunks c JOIN documentos d ON c.documento_id = d.id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1::vector ASC LIMIT 3
    """, str(pt_emb))

    en_results = await conn.fetch("""
        SELECT d.filename, d.tipo, d.numero, d.ano,
               1 - (c.embedding <=> $1::vector) as similarity
        FROM chunks c JOIN documentos d ON c.documento_id = d.id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1::vector ASC LIMIT 3
    """, str(en_emb))

    print(f"  PT: \"{pt_query}\"")
    for i, r in enumerate(pt_results, 1):
        print(f"    {i}. Sim={float(r['similarity']):.4f} | {r['tipo']} {r['numero']}/{r['ano']} ({r['filename']})")

    print(f"\n  EN: \"{en_query}\"")
    for i, r in enumerate(en_results, 1):
        print(f"    {i}. Sim={float(r['similarity']):.4f} | {r['tipo']} {r['numero']}/{r['ano']} ({r['filename']})")

    # Check overlap
    pt_files = {r['filename'] for r in pt_results}
    en_files = {r['filename'] for r in en_results}
    overlap = pt_files & en_files
    print(f"\n  Overlap PT/EN top-3: {len(overlap)}/3 docs em comum")
    if overlap:
        print(f"  ✅ Modelo multilíngue funciona! Mesmos docs para PT e EN.")
    else:
        print(f"  ℹ️  Resultados diferentes — normal se queries capturam nuances diferentes")

    await conn.close()
    print("\n" + "=" * 70)
    print("✅ TODOS OS TESTES CONCLUÍDOS")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
