# Ingestão e Busca Semântica com LangChain e PostgreSQL

Aplicação Python para ingerir um PDF local, salvar embeddings em PostgreSQL com pgVector e responder perguntas via CLI usando apenas o contexto recuperado do documento.

## Pré-requisitos

- Python 3.10 ou superior
- Docker
- Docker Compose
- API key do Google/Gemini
- Um arquivo `document.pdf` na raiz do projeto

## Configuração

Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Edite o `.env` e informe sua chave:

```env
GOOGLE_API_KEY=sua_api_key_google
```

As demais variáveis já vêm com valores padrão:

```env
PDF_PATH=document.pdf
POSTGRES_CONNECTION=postgresql+psycopg://postgres:postgres@localhost:5432/postgres
PGVECTOR_COLLECTION=documents
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite
EMBEDDING_BATCH_SIZE=10
EMBEDDING_BATCH_SLEEP_SECONDS=65
```

## Banco de dados

Suba o PostgreSQL com pgVector:

```bash
docker compose up -d
```

O serviço `bootstrap_vector_ext` do `docker-compose.yml` cria a extensão `vector` após o PostgreSQL ficar saudável.

## Ingestão do PDF

Coloque o arquivo `document.pdf` na raiz do projeto e execute:

```bash
python src/ingest.py
```

O script carrega o PDF com `PyPDFLoader`, divide o conteúdo em chunks de 1000 caracteres com overlap de 150, gera embeddings com `models/gemini-embedding-001` e grava os vetores no PostgreSQL. A ingestão é feita em lotes configuráveis por `EMBEDDING_BATCH_SIZE` e `EMBEDDING_BATCH_SLEEP_SECONDS` para respeitar limites de requisições da API.

## Chat via terminal

Depois da ingestão, rode:

```bash
python src/chat.py
```

Exemplo:

```text
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Perguntas fora do contexto devem receber exatamente:

```text
Não tenho informações necessárias para responder sua pergunta.
```

Para encerrar o chat, digite `sair`, `exit` ou `quit`.

## Busca semântica

O módulo `src/search.py` centraliza a configuração do vector store e usa:

```python
similarity_search_with_score(query, k=10)
```

Os 10 trechos mais relevantes são concatenados no contexto enviado para a LLM. O prompt instrui o modelo a responder somente com base nesse contexto, sem conhecimento externo, opiniões ou interpretações além do texto recuperado.
