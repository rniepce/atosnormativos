import asyncio
import asyncpg
async def run():
    conn = await asyncpg.connect("postgresql://postgres:pvjPoRKOeQOVmZNCulAVYXqpWmnefbsa@yamanote.proxy.rlwy.net:23504/railway")
    await conn.execute("DELETE FROM documentos WHERE filename IN ('lei_complementar_59_2001.txt', 'regimento_interno_tjmg.pdf')")
    print("Deleted.")
    await conn.close()
asyncio.run(run())
