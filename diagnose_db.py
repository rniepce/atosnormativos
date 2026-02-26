"""
Comprehensive database diagnostic script for the normative acts database.
Checks: document counts, chunk counts, embedding quality, metadata quality, coverage.
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import asyncpg

async def main():
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB")
        )
    except Exception as e:
        print(f"❌ DB Connection failed: {e}")
        return

    print("=" * 70)
    print("📊 DIAGNÓSTICO COMPLETO DO BANCO DE DADOS DE ATOS NORMATIVOS")
    print("=" * 70)

    # 1. Basic counts
    print("\n" + "─" * 50)
    print("1️⃣  CONTAGENS BÁSICAS")
    print("─" * 50)
    
    doc_count = await conn.fetchval("SELECT COUNT(*) FROM documentos")
    chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    print(f"  Documentos: {doc_count}")
    print(f"  Chunks: {chunk_count}")
    if doc_count > 0:
        print(f"  Média de chunks por documento: {chunk_count / doc_count:.1f}")

    # 2. Documents by type
    print("\n" + "─" * 50)
    print("2️⃣  DOCUMENTOS POR TIPO")
    print("─" * 50)
    
    rows = await conn.fetch("""
        SELECT tipo, COUNT(*) as qtd 
        FROM documentos 
        GROUP BY tipo 
        ORDER BY qtd DESC
    """)
    for row in rows:
        print(f"  {row['tipo']}: {row['qtd']}")

    # 3. Documents by year
    print("\n" + "─" * 50)
    print("3️⃣  DOCUMENTOS POR ANO")
    print("─" * 50)
    
    rows = await conn.fetch("""
        SELECT ano, COUNT(*) as qtd 
        FROM documentos 
        GROUP BY ano 
        ORDER BY ano DESC
    """)
    zero_year = 0
    for row in rows:
        if row['ano'] == 0:
            zero_year = row['qtd']
        else:
            print(f"  {row['ano']}: {row['qtd']} docs")
    if zero_year > 0:
        print(f"  ⚠️  Ano = 0 (não extraído): {zero_year} docs")

    # 4. Documents by status
    print("\n" + "─" * 50)
    print("4️⃣  DOCUMENTOS POR STATUS")
    print("─" * 50)
    
    rows = await conn.fetch("""
        SELECT status_vigencia, COUNT(*) as qtd 
        FROM documentos 
        GROUP BY status_vigencia 
        ORDER BY qtd DESC
    """)
    for row in rows:
        print(f"  {row['status_vigencia'] or 'NULL'}: {row['qtd']}")

    # 5. Embedding quality checks
    print("\n" + "─" * 50)
    print("5️⃣  QUALIDADE DOS EMBEDDINGS")
    print("─" * 50)
    
    # Check for NULL embeddings
    null_emb = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
    print(f"  Chunks sem embedding (NULL): {null_emb}")
    
    # Check embedding dimensions
    dim_check = await conn.fetchval("""
        SELECT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL LIMIT 1
    """)
    print(f"  Dimensão dos embeddings: {dim_check}")
    
    # Check if all embeddings have same dimension
    dim_variance = await conn.fetch("""
        SELECT vector_dims(embedding) as dim, COUNT(*) as qtd 
        FROM chunks 
        WHERE embedding IS NOT NULL 
        GROUP BY vector_dims(embedding)
    """)
    if len(dim_variance) > 1:
        print("  ⚠️  PROBLEMA: Embeddings com dimensões diferentes!")
        for row in dim_variance:
            print(f"      Dim {row['dim']}: {row['qtd']} chunks")
    else:
        print(f"  ✅ Todos os embeddings têm dimensão consistente ({dim_check})")

    # Check for zero/near-zero norm embeddings (bad embeddings)
    zero_norm = await conn.fetchval("""
        SELECT COUNT(*) FROM chunks 
        WHERE embedding IS NOT NULL 
        AND vector_norm(embedding) < 0.01
    """)
    print(f"  Chunks com embedding de norma ~0 (inválidos): {zero_norm}")

    # Sample embedding norms
    norms = await conn.fetch("""
        SELECT vector_norm(embedding) as norm 
        FROM chunks 
        WHERE embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 20
    """)
    norm_values = [float(r['norm']) for r in norms]
    if norm_values:
        print(f"  Norma dos embeddings (amostra de 20):")
        print(f"    Min: {min(norm_values):.4f}")
        print(f"    Max: {max(norm_values):.4f}")
        print(f"    Média: {sum(norm_values)/len(norm_values):.4f}")

    # 6. Content quality
    print("\n" + "─" * 50)
    print("6️⃣  QUALIDADE DO CONTEÚDO")
    print("─" * 50)
    
    # Very short chunks
    short_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE LENGTH(conteudo_texto) < 50")
    print(f"  Chunks muito curtos (<50 chars): {short_chunks}")
    
    # Very long chunks
    long_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE LENGTH(conteudo_texto) > 5000")
    print(f"  Chunks muito longos (>5000 chars): {long_chunks}")
    
    # Average chunk size
    avg_size = await conn.fetchval("SELECT AVG(LENGTH(conteudo_texto)) FROM chunks")
    print(f"  Tamanho médio dos chunks: {avg_size:.0f} chars")

    # Docs without chunks
    orphan_docs = await conn.fetchval("""
        SELECT COUNT(*) FROM documentos d 
        WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.documento_id = d.id)
    """)
    print(f"  Documentos sem nenhum chunk: {orphan_docs}")

    # 7. Metadata quality
    print("\n" + "─" * 50)
    print("7️⃣  QUALIDADE DOS METADADOS")
    print("─" * 50)
    
    no_tipo = await conn.fetchval("SELECT COUNT(*) FROM documentos WHERE tipo IS NULL OR tipo = 'Desconhecido'")
    no_numero = await conn.fetchval("SELECT COUNT(*) FROM documentos WHERE numero IS NULL OR numero = '0'")
    no_ano = await conn.fetchval("SELECT COUNT(*) FROM documentos WHERE ano IS NULL OR ano = 0")
    no_orgao = await conn.fetchval("SELECT COUNT(*) FROM documentos WHERE orgao IS NULL")
    
    print(f"  Sem tipo (ou 'Desconhecido'): {no_tipo} ({no_tipo*100/max(doc_count,1):.1f}%)")
    print(f"  Sem número (ou '0'): {no_numero} ({no_numero*100/max(doc_count,1):.1f}%)")
    print(f"  Sem ano (ou 0): {no_ano} ({no_ano*100/max(doc_count,1):.1f}%)")
    print(f"  Sem órgão: {no_orgao} ({no_orgao*100/max(doc_count,1):.1f}%)")

    # 8. Duplicate check
    print("\n" + "─" * 50)
    print("8️⃣  VERIFICAÇÃO DE DUPLICATAS")
    print("─" * 50)
    
    dupes = await conn.fetch("""
        SELECT filename, COUNT(*) as qtd 
        FROM documentos 
        GROUP BY filename 
        HAVING COUNT(*) > 1 
        ORDER BY qtd DESC 
        LIMIT 20
    """)
    if dupes:
        print(f"  ⚠️  {len(dupes)} arquivos duplicados encontrados!")
        for row in dupes[:10]:
            print(f"    {row['filename']}: {row['qtd']}x")
    else:
        print("  ✅ Nenhum arquivo duplicado")

    # 9. Sample search test (self-similarity with stored embeddings)
    print("\n" + "─" * 50)
    print("9️⃣  TESTE DE BUSCA (auto-similaridade)")
    print("─" * 50)
    
    # Pick a random chunk and find its nearest neighbors
    sample = await conn.fetchrow("""
        SELECT c.id, c.conteudo_texto, d.tipo, d.numero, d.ano, d.filename
        FROM chunks c 
        JOIN documentos d ON c.documento_id = d.id
        WHERE embedding IS NOT NULL AND LENGTH(c.conteudo_texto) > 200
        ORDER BY RANDOM() LIMIT 1
    """)
    
    if sample:
        print(f"  Chunk de referência: {sample['tipo']} {sample['numero']}/{sample['ano']} ({sample['filename']})")
        print(f"  Texto (trecho): {sample['conteudo_texto'][:150]}...")
        
        neighbors = await conn.fetch("""
            SELECT c.id, d.filename, d.tipo, d.numero, d.ano,
                   1 - (c.embedding <=> (SELECT embedding FROM chunks WHERE id = $1)) as similarity
            FROM chunks c
            JOIN documentos d ON c.documento_id = d.id
            WHERE c.id != $1 AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> (SELECT embedding FROM chunks WHERE id = $1) ASC
            LIMIT 5
        """, sample['id'])
        
        print(f"\n  Top 5 vizinhos mais próximos:")
        for i, n in enumerate(neighbors, 1):
            print(f"    {i}. Similaridade: {float(n['similarity']):.4f} | {n['tipo']} {n['numero']}/{n['ano']} ({n['filename']})")

    # 10. Documents by orgao
    print("\n" + "─" * 50)
    print("🔟  DOCUMENTOS POR ÓRGÃO")
    print("─" * 50)
    
    rows = await conn.fetch("""
        SELECT orgao, COUNT(*) as qtd 
        FROM documentos 
        GROUP BY orgao 
        ORDER BY qtd DESC
    """)
    for row in rows:
        print(f"  {row['orgao'] or 'NULL'}: {row['qtd']}")

    # 11. Check created_at column exists and recency
    print("\n" + "─" * 50)
    print("1️⃣1️⃣  DATA DE INGESTÃO")
    print("─" * 50)
    
    try:
        recent = await conn.fetch("""
            SELECT data_upload, COUNT(*) as qtd 
            FROM documentos
            GROUP BY DATE(data_upload)
            ORDER BY data_upload DESC
            LIMIT 10
        """)
        for row in recent:
            print(f"  {row['data_upload']}: {row['qtd']} docs ingeridos")
    except Exception as e:
        print(f"  Erro ao verificar datas: {e}")

    # 12. Summary of potential issues
    print("\n" + "=" * 70)
    print("📋 RESUMO DOS PROBLEMAS ENCONTRADOS")
    print("=" * 70)
    
    issues = []
    
    if doc_count < 1000:
        issues.append(f"⚠️  POUCOS DOCUMENTOS: Apenas {doc_count} documentos no banco (havia ~13.000 arquivos para processar)")
    
    if null_emb > 0:
        issues.append(f"⚠️  {null_emb} chunks sem embedding")
    
    if zero_norm > 0:
        issues.append(f"⚠️  {zero_norm} chunks com embedding inválido (norma ~0)")
    
    if orphan_docs > 0:
        issues.append(f"⚠️  {orphan_docs} documentos sem chunks")
    
    if no_tipo > doc_count * 0.1:
        issues.append(f"⚠️  {no_tipo} documentos sem tipo classificado")
    
    if no_ano > doc_count * 0.1:
        issues.append(f"⚠️  {no_ano} documentos sem ano extraído")
    
    if dupes:
        issues.append(f"⚠️  {len(dupes)} arquivos com duplicatas no banco")
    
    # CRITICAL: Embedding model mismatch
    issues.append("🔴 CRÍTICO: Incompatibilidade de modelos de embedding!")
    issues.append("   → Ingestão usou: BAAI/bge-large-en-v1.5 (local, inglês)")
    issues.append("   → Busca usa: Azure text-embedding-3-large (cloud, multilíngue)")
    issues.append("   → Apesar de ambos terem 1024 dimensões, os espaços vetoriais são COMPLETAMENTE DIFERENTES")
    issues.append("   → Isso significa que a busca por similaridade NÃO FUNCIONA corretamente!")
    
    for issue in issues:
        print(f"  {issue}")
    
    if not issues:
        print("  ✅ Nenhum problema crítico encontrado")

    await conn.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
