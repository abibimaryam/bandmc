# import os
# from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from embedding_functiom import embedding_function
# from config import CHROMA_PATH, model

# # Новый промпт для проверки несоответствий (только выводим несоответствия)
# PROMPT_TEMPLATE = """
# Ниже приведён контекст из медицинской базы данных:

# {context}

# Пользователь сделал следующее утверждение:

# {question}

# Проанализируй утверждение в контексте и найди ВСЕ несоответствия или ошибки относительно предоставленной информации.

# Выведи список только тех пунктов, где утверждение НЕ СООТВЕТСТВУЕТ данным из контекста, с кратким объяснением.

# Если несоответствий нет, ответь "Несоответствий не обнаружено."
# """

# def extract_key_entities(user_input: str) -> str:
#     prompt = f"""
# Извлеки ключевые элементы из следующего запроса, такие как:

# - торговое название лекарства,
# - международное непатентованное название (если есть),
# - форма выпуска,
# - дозировка,
# - страна-производитель,
# - фирма-производитель,

# Верни их через запятую — одной строкой. Если ничего не найдено, верни "None".

# Запрос: "{user_input}"
# """
#     response = model.invoke(prompt).strip()
#     print(response)
#     return response if response.lower() != "none" else None


# def query_rag(user_input: str):
#     key_phrases = extract_key_entities(user_input)
#     if not key_phrases:
#         print("⚠️ Не удалось выделить ключевые элементы. Выполняется общий поиск.")
#         query_text = user_input
#     else:
#         print(f"🧷 Извлечены ключевые элементы: {key_phrases}")
#         query_text = key_phrases

#     db = Chroma(
#         collection_name="meds",
#         persist_directory=CHROMA_PATH,
#         embedding_function=embedding_function()
#     )

#     print(f"🔍 Документов в базе: {db._collection.count()}")
#     print("🔎 Выполняется поиск по векторной базе...")
#     results = db.similarity_search_with_score(query_text, k=20)

#     if not results:
#         print("❌ Ничего не найдено.")
#         return "Ничего не найдено."

#     print(f"\n📄 Найдено {len(results)} релевантных документов:\n")
#     for i, (doc, score) in enumerate(results):
#         print(f"🔹 Документ {i+1} | Название: {doc.metadata.get('торговое_название', 'Неизвестно')} | Score: {score:.4f}")
#         print(doc.page_content[:300])
#         print("------")

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=user_input)

#     response_text = model.invoke(prompt).strip()

#     sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results]
#     print(f"\n🧠 Ответ с несоответствиями:\n{response_text}")
#     print(f"\n📚 Источники: {sources}")

#     return response_text


# def run_rag():
#     if os.path.exists(CHROMA_PATH):
#         print("📁 Файлы в chroma_db:")
#         print(os.listdir(CHROMA_PATH))
#     else:
#         print(f"❗ Папка {CHROMA_PATH} не найдена!")

#     vectordb = Chroma(
#         collection_name="meds",
#         embedding_function=embedding_function(),
#         persist_directory=CHROMA_PATH,
#     )
#     print(f"📊 Документов в базе: {vectordb._collection.count()}")

#     query_text = input("\n❓ Введите ваше утверждение: ")
#     query_rag(query_text)


# if __name__ == "__main__":
#     run_rag()


import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_functiom import embedding_function
from config import CHROMA_PATH, model
from duckduckgo_search import DDGS

# Новый промпт для проверки несоответствий (только выводим несоответствия)
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

# Веб-поиск через DuckDuckGo (использует DDGS, duckduckgo_search)

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
    if not key_phrases:
        print("⚠️ Не удалось выделить ключевые элементы. Выполняется общий поиск.")
        query_text = user_input
        filter_name = None
    else:
        print(f"🧷 Извлечены ключевые элементы: {key_phrases}")
        query_text = key_phrases
        # Предположим, что название лекарства — первое слово ключевых сущностей
        filter_name = key_phrases.split(",")[0].strip().lower()

    # Поиск в локальной базе
    db = Chroma(
        collection_name="meds",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )

    print(f"🔍 Документов в базе: {db._collection.count()}")
    print("🔎 Выполняется поиск по векторной базе...")
    results = db.similarity_search_with_score(query_text, k=20)

    # Фильтрация документов по названию лекарства
    if filter_name:
        filtered_results = []
        for doc, score in results:
            title = doc.metadata.get("торговое_название", "").lower()
            if filter_name in title:
                filtered_results.append((doc, score))
        results = filtered_results

    if not results:
        print("❌ Ничего не найдено в локальной базе по фильтру.")
        local_context = ""
    else:
        print(f"\n📄 Найдено {len(results)} релевантных документов по фильтру '{filter_name}':\n")
        for i, (doc, score) in enumerate(results):
            print(f"🔹 Документ {i+1} | Название: {doc.metadata.get('торговое_название', 'Неизвестно')} | Score: {score:.4f}")
            print(doc.page_content[:300])
            print("------")
        local_context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Поиск в интернете
    print("🌐 Выполняется веб-поиск...")
    web_results = search_tool(user_input)
    print(f"🌐 Результаты веб-поиска (первые 500 символов):\n{web_results[:500]}...\n")

    # Объединяем контексты
    combined_context = f"Локальная база данных:\n{local_context}\n\nИнтернет данные:\n{web_results}"

    # Формируем финальный промпт с расширенным контекстом
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=combined_context, question=user_input, key_entities=key_phrases or "None")

    # Запрос к модели
    response_text = model.invoke(prompt).strip()

    sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results] if results else []
    print(f"\n🧠 Ответ с несоответствиями:\n{response_text}")
    print(f"\n📚 Источники: локальная база - {sources}, интернет - DuckDuckGo")

    return response_text



def run_rag():
    if os.path.exists(CHROMA_PATH):
        print("📁 Файлы в chroma_db:")
        print(os.listdir(CHROMA_PATH))
    else:
        print(f"❗ Папка {CHROMA_PATH} не найдена!")

    vectordb = Chroma(
        collection_name="meds",
        embedding_function=embedding_function(),
        persist_directory=CHROMA_PATH,
    )
    print(f"📊 Документов в базе: {vectordb._collection.count()}")

    query_text = input("\n❓ Введите ваше утверждение: ")
    query_rag_with_web_search(query_text)


if __name__ == "__main__":
    run_rag()
