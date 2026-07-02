from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from search import TOP_K, format_context, load_environment, require_google_api_key, search_documents


FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."
EXIT_COMMANDS = {"sair", "exit", "quit"}
DEFAULT_CHAT_MODEL = "gemini-2.5-flash-lite"


def build_prompt(context: str, question: str) -> str:
    return f"""CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "{FALLBACK_ANSWER}"
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "{FALLBACK_ANSWER}"

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "{FALLBACK_ANSWER}"

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "{FALLBACK_ANSWER}"

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def create_llm() -> ChatGoogleGenerativeAI:
    require_google_api_key()
    model = os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def normalize_answer(answer: str) -> str:
    cleaned_answer = answer.strip().strip('"')

    if FALLBACK_ANSWER in cleaned_answer:
        return FALLBACK_ANSWER

    return cleaned_answer


def answer_question(question: str) -> str:
    results = search_documents(question, k=TOP_K)
    context = format_context(results)

    if not context.strip():
        return FALLBACK_ANSWER

    llm = create_llm()
    response = llm.invoke(build_prompt(context=context, question=question))
    content = response.content

    if isinstance(content, str):
        return normalize_answer(content)

    return normalize_answer(str(content))


def main() -> None:
    load_environment()
    print("Faça sua pergunta. Digite 'sair', 'exit' ou 'quit' para encerrar.")

    while True:
        question = input("\nPERGUNTA: ").strip()

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando.")
            break

        try:
            print(f"RESPOSTA: {answer_question(question)}")
        except Exception as error:
            print(f"Erro: {error}")


if __name__ == "__main__":
    main()
