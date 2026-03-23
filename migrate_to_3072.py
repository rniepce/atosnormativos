"""
Migração: vector(1024) → vector(3072)
======================================
Altera a coluna de embedding no banco para suportar 3072 dimensões
(máxima qualidade do text-embedding-3-large).

O que este script faz:
1. Zera todos os embeddings existentes (eles serão regenerados pelo reembed_azure.py)
2. Altera o tipo da coluna de vector(1024) para vector(3072)
3. Recria o índice HNSW com a nova dimensão
4. Mantém todos os documentos e textos intactos

IMPORTANTE: Rode este script ANTES de rodar o reembed_azure.py.
Após este script, rode: python3 reembed_azure.py --rechunk

Usage:
    python3 migrate_to_3072.py             # Executa a migração
    python3 migrate_to_3072.py --dry-run  # Simula sem alterar
"""
import asyncio
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import asyncpg

load_dotenv(Path(__file__).parent / ".env")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Simula sem alterar o banco")
    parser.add_argument("--yes", action="store_true", help="Confirma automaticamente sem prompt interativo")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 MIGRAÇÃO: vector(1024) → vector(3072)")
    print("=" * 60)

    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB"),
        )
        print("✅ Conectado ao banco\n")
    except Exception as e:
        print(f"❌ Falha na conexão: {e}")
        sys.exit(1)

    # Estado atual
    docs  = await conn.fetchval("SELECT COUNT(*) FROM documentos")
    chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    dim_row = await conn.fetchrow("""
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
    """)
    current_dim = (dim_row["atttypmod"] - 4) if dim_row and dim_row["atttypmod"] > 0 else "?"

    print(f"📊 Estado atual:")
    print(f"   Documentos: {docs}")
    print(f"   Chunks:     {chunks}")
    print(f"   Dimensão atual: {current_dim}")
    print()

    if current_dim == 3072:
        print("✅ Banco já está em 3072 dimensões. Nada a fazer.")
        await conn.close()
        return

    if args.dry_run:
        print("[DRY-RUN] Executaria:")
        print("  1. UPDATE chunks SET embedding = NULL")
        print("  2. DROP INDEX chunks_embedding_idx (HNSW)")
        print("  3. ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(3072)")
        print("  4. CREATE INDEX USING hnsw (embedding vector_cosine_ops)")
        print("\n🏁 [DRY-RUN] Concluído. Nenhuma alteração feita.")
        await conn.close()
        return

    print("⚠️  Esta operação irá:")
    print("   • Zerar todos os embeddings (serão regenerados pelo reembed_azure.py)")
    print("   • Alterar a coluna embedding de vector(1024) para vector(3072)")
    print("   • Recriar o índice HNSW")
    print("   • NÃO apaga documentos nem textos dos chunks")
    print()
    if args.yes:
        print("✅ Confirmado via --yes\n")
    else:
        confirm = input("Confirmar? (digite 'sim' para prosseguir): ").strip().lower()
        if confirm != "sim":
            print("Cancelado.")
            await conn.close()
            return

    print()

    try:
        # Passo 1: Zera embeddings (necessário para alterar o tipo da coluna)
        print("1️⃣  Zerando embeddings existentes...")
        await conn.execute("UPDATE chunks SET embedding = NULL")
        print(f"   ✅ {chunks} embeddings zerados\n")

        # Passo 2: Remove índice HNSW (não é possível alterar coluna com índice ativo)
        print("2️⃣  Removendo índice HNSW antigo...")
        await conn.execute("""
            DO $$
            DECLARE idx_name text;
            BEGIN
                SELECT indexname INTO idx_name
                FROM pg_indexes
                WHERE tablename = 'chunks' AND indexdef ILIKE '%hnsw%'
                LIMIT 1;
                IF idx_name IS NOT NULL THEN
                    EXECUTE 'DROP INDEX IF EXISTS ' || idx_name;
                END IF;
            END $$;
        """)
        print("   ✅ Índice removido\n")

        # Passo 3: Altera tipo da coluna
        print("3️⃣  Alterando coluna para vector(3072)...")
        await conn.execute("""
            ALTER TABLE chunks
            ALTER COLUMN embedding TYPE vector(3072)
            USING embedding::vector(3072)
        """)
        print("   ✅ Coluna alterada\n")

        # Passo 4: Recria índice HNSW
        # m=16 e ef_construction=128 são bons defaults para datasets desta escala (~55k chunks)
        print("4️⃣  Criando novo índice HNSW (vector_cosine_ops, m=16, ef_construction=128)...")
        await conn.execute("""
            CREATE INDEX ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 128)
        """)
        print("   ✅ Índice criado\n")

        # Confirmação final
        new_dim_row = await conn.fetchrow("""
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
        """)
        new_dim = (new_dim_row["atttypmod"] - 4) if new_dim_row else "?"

        print("=" * 60)
        print("🎉 MIGRAÇÃO CONCLUÍDA!")
        print(f"   Dimensão: {current_dim} → {new_dim}")
        print(f"   Documentos preservados: {docs}")
        print(f"   Chunks (textos preservados, embeddings zerados): {chunks}")
        print()
        print("▶️  Próximo passo:")
        print("   python3 reembed_azure.py --rechunk")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Erro durante a migração: {e}")
        print("   O banco pode estar em estado inconsistente.")
        print("   Verifique e execute novamente ou rode apply_migrations.py para recriar do zero.")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
