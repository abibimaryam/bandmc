# import os
# from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from embedding_functiom import embedding_function
# from config import CHROMA_PATH,model



# CHROMA_PATH = "chroma_db"

# PROMPT_TEMPLATE = """
# Ответьте на следующий вопрос, используя только приведённый ниже контекст:

# {context}

# Вопрос: {question}
# """

# def query_rag(query_text: str):
#     db = Chroma(
#         collection_name="meds",
#         persist_directory=CHROMA_PATH,
#         embedding_function=embedding_function()
#     )

#     print(f"🔍 Документов в базе: {db._collection.count()}")
#     print("🔎 Выполняется поиск по векторной базе...")
#     results = db.similarity_search_with_score(query_text, k=10)

#     print(f"\n📄 Найдено {len(results)} релевантных документов:\n")
#     for i, (doc, score) in enumerate(results):
#         print(f"🔹 Документ {i+1} | Название: {doc.metadata.get('торговое_название', 'Неизвестно')} | Score: {score:.4f}")
#         print(doc.page_content[:300])
#         print("------")

#     if not results:
#         print("❌ Ничего не найдено.")
#         return "Ничего не найдено."

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results]
#     print(f"\n🧠 Ответ:\n{response_text}")
#     print(f"\n📚 Источники: {sources}")

#     return response_text


# def run_rag():
#     # Проверка наличия базы
#     if os.path.exists(CHROMA_PATH):
#         print("📁 Файлы в chroma_db:")
#         print(os.listdir(CHROMA_PATH))
#     else:
#         print(f"❗ Папка {CHROMA_PATH} не найдена!")


#     # Проверка количества документов
#     vectordb = Chroma(
#         collection_name="meds",
#         embedding_function=embedding_function(),
#         persist_directory=CHROMA_PATH,
#     )
#     print(f"📊 Документов в базе: {vectordb._collection.count()}")

#     # Запрос пользователя
#     query_text = input("\n❓ Введите ваш вопрос: ")
#     query_rag(query_text)


# if __name__ == "__main__":
#     run_rag()


import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_functiom import embedding_function
from config import CHROMA_PATH, model


PROMPT_TEMPLATE = """
Ответьте на следующий вопрос, используя только приведённый ниже контекст:

{context}

Вопрос: {question}
"""

# 📌 Извлечение ключевых сущностей
def extract_key_entities(user_input: str) -> str:
    prompt = f"""
Извлеки ключевые элементы из следующего запроса, такие как:

- торговое название лекарства,
- международное непатентованное название (если есть),
- форма выпуска,
- дозировка,
- страна-производитель,
- фирма-производитель,


Верни их через запятую — одной строкой. Если ничего не найдено, верни "None".

Запрос: "{user_input}"
"""
    response = model.invoke(prompt).strip()
    print(response)
    return response if response.lower() != "none" else None


def query_rag(user_input: str):
    key_phrases = extract_key_entities(user_input)
    if not key_phrases:
        print("⚠️ Не удалось выделить ключевые элементы. Выполняется общий поиск.")
        query_text = user_input
    else:
        print(f"🧷 Извлечены ключевые элементы: {key_phrases}")
        query_text = key_phrases

    db = Chroma(
        collection_name="meds",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )

    print(f"🔍 Документов в базе: {db._collection.count()}")
    print("🔎 Выполняется поиск по векторной базе...")
    results = db.similarity_search_with_score(query_text, k=20)

    if not results:
        print("❌ Ничего не найдено.")
        return "Ничего не найдено."

    print(f"\n📄 Найдено {len(results)} релевантных документов:\n")
    for i, (doc, score) in enumerate(results):
        print(f"🔹 Документ {i+1} | Название: {doc.metadata.get('торговое_название', 'Неизвестно')} | Score: {score:.4f}")
        print(doc.page_content[:300])
        print("------")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_input)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results]
    print(f"\n🧠 Ответ:\n{response_text}")
    print(f"\n📚 Источники: {sources}")

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

    query_text = input("\n❓ Введите ваш вопрос: ")
    query_rag(query_text)


if __name__ == "__main__":
    run_rag()
