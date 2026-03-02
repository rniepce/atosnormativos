# Documentação Técnica - Atos Normativos

## 1. Visão Geral

O sistema **Atos Normativos** é uma aplicação de backend projetada para gerenciar e consultar documentos normativos. A sua principal função é realizar a ingestão de documentos em formato PDF, processá-los, armazenar o conteúdo em um banco de dados vetorial e disponibilizar uma interface de busca semântica através de uma API.

O propósito de negócio é facilitar o acesso e a busca inteligente em um grande volume de documentos, permitindo que os usuários encontrem informações relevantes de forma rápida e precisa, utilizando linguagem natural. O sistema parece ser focado em documentos jurídicos ou regulatórios, como o "Regimento Interno.pdf" encontrado no repositório.

## 2. Stack Tecnológica

A seleção de tecnologias indica um foco em desenvolvimento moderno de aplicações Python, com ênfase em performance (assincronicidade) e integração com serviços de nuvem para IA e armazenamento.

| Tecnologia | Categoria | Relevância no Projeto |
| :--- | :--- | :--- |
| **Python** | Linguagem | Linguagem principal do backend, escolhida por seu ecossistema robusto para IA e desenvolvimento web. |
| **FastAPI** | Framework Web | Framework de alta performance para a criação da API, com suporte nativo a operações assíncronas. |
| **Uvicorn** | Servidor ASGI | Utilizado para rodar a aplicação FastAPI, garantindo a execução de código assíncrono. |
| **PostgreSQL (com pgvector)** | Banco de Dados | Armazena os dados da aplicação. A extensão `pgvector` é crucial para a busca de similaridade em embeddings vetoriais. |
| **SQLAlchemy** | ORM | Facilita a interação com o banco de dados PostgreSQL de forma estruturada e segura. |
| **Docker** | Containerização | Utilizado para empacotar e distribuir a aplicação, garantindo um ambiente de execução consistente. |
| **Google Cloud (Storage, AI Platform)** | Serviços de Nuvem | Utilizado para armazenamento de arquivos (PDFs) e para acessar modelos de IA (possivelmente para geração de embeddings). |
| **LangChain** | Framework de IA | Usado para orquestrar interações com modelos de linguagem (LLMs), como a divisão de texto. |
| **PyMuPDF (fitz)** | Biblioteca PDF | Essencial para a extração de texto e metadados dos documentos PDF ingeridos. |

## 3. Arquitetura

O sistema adota uma arquitetura **Monolítica com Integração de Microsserviços de Nuvem**.

- **Aplicação Monolítica:** O núcleo da aplicação é um único serviço de backend escrito em Python com FastAPI. Ele centraliza as responsabilidades de ingestão, processamento e consulta.

- **Serviços de Nuvem:** A aplicação delega tarefas especializadas a serviços externos, caracterizando uma abordagem híbrida:
  - **Google Cloud Storage:** Atua como um repositório de arquivos (Data Lake) para os documentos PDF.
  - **Google AI Platform (Vertex AI):** Utilizado para tarefas de IA, como a geração de embeddings a partir do texto extraído.
  - **Banco de Dados como Serviço:** O PostgreSQL provavelmente é hospedado em uma plataforma de nuvem como Heroku ou Railway.

A estrutura do diretório `src/` sugere uma tentativa de organização por responsabilidades, alinhada com princípios de design como o **Domain-Driven Design (DDD)** e **Clean Architecture**, com camadas para `api`, `services`, `repositories`, e `models`.

## 4. Fluxo de Dados: Ingestão de um Documento

O processo crítico de ingestão de um novo ato normativo segue as seguintes etapas:

1.  **Upload:** Um usuário (ou sistema) envia um arquivo PDF através de um endpoint da API (`/upload`).
2.  **Armazenamento Temporário:** A API FastAPI recebe o arquivo e o salva no Google Cloud Storage.
3.  **Processamento Assíncrono:** Uma tarefa de background é iniciada.
    -   O texto do PDF é extraído usando a biblioteca `PyMuPDF`.
    -   O texto é dividido em "chunks" (pedaços menores) pela `langchain_text_splitters`.
4.  **Geração de Embeddings:** Para cada "chunk" de texto, o sistema faz uma chamada a um serviço de IA (Google AI Platform ou OpenAI) para gerar um vetor de embedding.
5.  **Persistência:** O texto original, o "chunk" e o seu vetor de embedding correspondente são armazenados no banco de dados PostgreSQL com a extensão `pgvector`.
6.  **Disponibilização:** Uma vez que os dados estão no banco, eles se tornam disponíveis para consulta via API de busca.

## 5. Guia de Setup Local

Para executar o projeto localmente, é necessário ter o Docker e o Docker Compose instalados.

1.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/rniepce/atosnormativos.git
    cd atosnormativos
    ```

2.  **Configurar Variáveis de Ambiente:**
    Crie um arquivo `.env` na raiz do projeto. Este arquivo deve conter as credenciais de acesso aos serviços de nuvem e ao banco de dados.

    ```dotenv name=.env
    # Exemplo de variáveis de ambiente
    DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/atos_normativos_db"
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/gcp-credentials.json"
    GCS_BUCKET_NAME="seu-bucket-name"
    # Adicione outras variáveis necessárias (ex: chaves de API da OpenAI/Vertex AI)
    ```

3.  **Construir e Iniciar os Containers:**
    O `docker-compose.yml` irá orquestrar o serviço da aplicação e o banco de dados.

    ```bash
    docker-compose up --build
    ```

4.  **Aplicar Migrações (se houver):**
    O repositório contém um script `apply_migrations.py`. Se houver migrações de banco de dados no diretório `db/migrations`, elas precisam ser aplicadas.

    ```bash
    docker-compose exec backend python apply_migrations.py
    ```

5.  **Acessar a API:**
    A API estará disponível em `http://localhost:8000`. A documentação interativa (Swagger) pode ser acessada em `http://localhost:8000/docs`.

## 6. Principais Módulos e Classes

A estrutura de diretórios principal é a seguinte:

```
.
├── src/
│   ├── api/
│   │   └── routes/      # Endpoints da API (FastAPI)
│   ├── core/
│   │   └── settings.py  # Configurações da aplicação
│   ├── models/
│   │   └── document.py  # Modelos de dados (Pydantic/SQLAlchemy)
│   ├── repositories/
│   │   └── document.py  # Lógica de acesso ao banco de dados
│   └── services/
│       └── document.py  # Regras de negócio e orquestração
├── db/
│   └── migrations/      # Scripts de migração do banco de dados
├── scripts/             # Scripts utilitários (ex: reembed_azure.py)
├── Dockerfile.backend
└── docker-compose.yml
```

-   **`src/api/`**: Responsável por definir os endpoints da API, receber as requisições HTTP e retornar as respostas. Utiliza o FastAPI para roteamento.
-   **`src/services/`**: Contém a lógica de negócio principal. Orquestra as chamadas aos repositórios e serviços externos (IA, Cloud Storage) para executar tarefas como a ingestão de documentos.
-   **`src/repositories/`**: Camada de abstração do banco de dados. Isola a lógica de consulta e persistência de dados, utilizando SQLAlchemy para interagir com o PostgreSQL.
-   **`src/models/`**: Define as estruturas de dados da aplicação, como os schemas Pydantic para validação de dados da API e os modelos SQLAlchemy para mapeamento do banco.
-   **`src/core/`**: Módulo para configurações centrais da aplicação, como a leitura de variáveis de ambiente.
-   **`scripts/`**: Pasta com scripts para tarefas de manutenção e diagnóstico, como `diagnose_db.py` e `reembed_azure.py`, que sugerem a capacidade de re-processar documentos com diferentes modelos de embedding.

## 7. Pontos de Atenção e Recomendações

-   **Dívida Técnica:** O repositório contém múltiplos scripts de teste e depuração (`test_*.py`, `debug_*.py`) na raiz do projeto. Recomenda-se movê-los para um diretório `tests/` e integrá-los a um framework de testes como o `pytest`.
-   **Gerenciamento de Dependências:** O arquivo `requirements.txt` não especifica as versões das bibliotecas. Para garantir a reprodutibilidade do ambiente, é altamente recomendável o uso de uma ferramenta como `Poetry` ou `pip-tools` para fixar as versões das dependências.
-   **Segurança:** O arquivo `Regimento Interno.pdf` está commitado no repositório. Documentos e dados sensíveis não devem ser armazenados no Git. O uso do Google Cloud Storage já é um bom padrão, e o arquivo no repositório deveria ser removido.
-   **Flexibilidade de Modelos de IA:** A presença do script `reembed_azure.py` sugere que o sistema pode ter sido adaptado para usar embeddings da Azure, além do Google. Isso é positivo, mas a lógica de seleção do provedor de embedding poderia ser abstraída em uma camada de serviço para facilitar a troca e o teste.
