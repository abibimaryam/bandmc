import os
import re
import pandas as pd
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_functiom import embedding_function
from config import CHROMA_PATH, model
from duckduckgo_search import DDGS
from docx import Document
import pdfplumber

SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".csv", ".xlsx"]

PROMPT_TEMPLATE = """
Ниже приведён контекст из медицинской базы данных и интернета:

{context}

Пользователь сделал следующее утверждение:

{question}

В утверждении указаны следующие ключевые элементы: {key_entities}

Проанализируй утверждение в контексте и найди ВСЕ несоответствия или ошибки, особенно проверь, совпадают ли производитель и страна-производитель с базой данных и интернет-источниками.

Выведи список только тех пунктов, где утверждение НЕ СООТВЕТСТВУЕТ данным из контекста, с кратким объяснением.

Если несоответствий нет, ответь "Несоответствий не обнаружено."
"""

# --- Веб-поиск
def search_tool(query: str) -> str:
    with DDGS() as ddgs:
        results = [r["body"] for r in ddgs.text(query, max_results=3)]
    return "\n\n".join(results)


def extract_key_entities(user_input: str) -> str:
    prompt = f"""
Извлеки ключевые элементы из следующего запроса, такие как:

- торговое название лекарства,
- международное непатентованное название (если есть),

Верни их через запятую — одной строкой. Если ничего не найдено, верни "None".

Запрос: "{user_input}"
"""
    response = model.invoke(prompt).strip()
    print(f"🔑 Извлечённые ключевые элементы: {response}")
    return response if response.lower() != "none" else None


def query_rag_with_web_search(user_input: str):
    key_phrases = extract_key_entities(user_input)
    query_text = key_phrases or user_input
    filter_name = key_phrases.split(",")[0].strip().lower() if key_phrases else None

    db = Chroma(
        collection_name="meds",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )
    results = db.similarity_search_with_score(query_text, k=20)

    if filter_name:
        results = [(doc, score) for doc, score in results
                   if filter_name in doc.metadata.get("торговое_название", "").lower()]

    local_context = "\n\n---\n\n".join([doc.page_content for doc, _ in results]) if results else ""
    web_results = search_tool(user_input)

    combined_context = f"Локальная база данных:\n{local_context}\n\nИнтернет данные:\n{web_results}"

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=combined_context,
        question=user_input,
        key_entities=key_phrases or "None"
    )

    response_text = model.invoke(prompt).strip()
    return response_text


# --- Функции чтения файлов
def read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def read_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(filepath: str) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def read_excel_csv(filepath: str) -> str:
    ext = os.path.splitext(filepath)[-1].lower()
    df = pd.read_excel(filepath) if ext == ".xlsx" else pd.read_csv(filepath)
    return "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))


# --- Универсальный загрузчик текста
def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".txt":
        return read_txt(filepath)
    elif ext == ".docx":
        return read_docx(filepath)
    elif ext == ".pdf":
        return read_pdf(filepath)
    elif ext in [".csv", ".xlsx"]:
        return read_excel_csv(filepath)
    else:
        raise ValueError(f"❌ Формат {ext} не поддерживается. Поддерживаются: {', '.join(SUPPORTED_EXTENSIONS)}")


def split_document_into_statements(text: str) -> list:
    # Разделение по строкам или по предложениям
    return [s.strip() for s in re.split(r'\n+|(?<=[.!?])\s+', text) if s.strip()]


# --- Главная функция обработки документа
def process_document(filepath: str):
    if not os.path.exists(filepath):
        print(f"❌ Файл {filepath} не найден.")
        return

    print(f"📂 Загрузка файла: {filepath}")
    try:
        text = extract_text_from_file(filepath)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    statements = split_document_into_statements(text)
    print(f"📄 Найдено {len(statements)} утверждений.\n")

    all_results = []
    for i, statement in enumerate(statements, start=1):
        print(f"\n🔍 Утверждение {i}:\n{statement}")
        result = query_rag_with_web_search(statement)
        all_results.append((statement, result))

    print("\n🧾 Итоговый отчёт по несоответствиям:\n")
    for i, (stmt, mismatch) in enumerate(all_results, start=1):
        print(f"{i}. 📌 {stmt}")
        print(f"   ❗ Проверка: {mismatch}")
        print("------")


# --- Точка входа
if __name__ == "__main__":
    FILEPATH = "/home/maryam/Документы/projects/bandmc/test_data/test.txt"  # путь к файлу
    process_document(FILEPATH)
