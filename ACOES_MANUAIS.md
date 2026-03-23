# Ações Manuais Pendentes

> Gerado após code review em 22/03/2026. Estas ações **não foram automatizadas** e precisam ser feitas manualmente.

---

## [ ] 1. Rotacionar senha do PostgreSQL (Railway)

1. Acesse o dashboard do Railway → seu projeto → banco de dados
2. Clique em **Regenerate credentials**
3. Atualize o `.env` local com a nova senha
4. Atualize a variável `DATABASE_URL` / `POSTGRES_PASSWORD` nas **Railway Variables** (Settings → Variables)

---

## [ ] 2. Rotacionar chave da Azure OpenAI

1. Acesse o **Azure Portal** → seu recurso de Azure OpenAI
2. Vá em **Keys and Endpoint**
3. Clique em **Regenerate Key 1**
4. Atualize `AZURE_API_KEY` no `.env` local e nas variáveis do Railway

---

## [ ] 3. Rotacionar chave do Gemini API

1. Acesse **Google AI Studio** (aistudio.google.com) → **Get API key**
2. Delete a chave antiga
3. Crie uma nova chave
4. Atualize `GEMINI_API_KEY` no `.env` local e nas variáveis do Railway

---

## [ ] 4. Rotacionar chave da Amazonia IA

1. Acesse o portal da Amazonia IA
2. Regenere sua chave de API
3. Atualize a variável correspondente no `.env` local e nas variáveis do Railway

---

## [ ] 5. Gerar nova chave de upload

A chave atual (`proto-dev-key-change-in-production`) é fraca e previsível.

Gere uma chave forte com:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Atualize `UPLOAD_API_KEY` no `.env` local e nas variáveis do Railway.

---

## [ ] 6. Definir variável SOURCE_DIR no ambiente

Os scripts de ingestion agora usam `os.getenv("SOURCE_DIR")` em vez de caminhos hardcoded.

Defina a variável no seu ambiente local (`.env` ou shell):

```bash
SOURCE_DIR=/caminho/para/seus/documentos
```

---

## Checklist rápido

- [ ] PostgreSQL — senha rotacionada no Railway
- [ ] Azure OpenAI — chave regenerada
- [ ] Gemini — chave regenerada
- [ ] Amazonia IA — chave regenerada
- [ ] Upload key — nova chave gerada e atualizada
- [ ] SOURCE_DIR — variável configurada no ambiente
