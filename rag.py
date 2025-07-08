import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_functiom import embedding_function
from config import CHROMA_PATH,model



CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
Ответьте на следующий вопрос, используя только приведённый ниже контекст:

{context}

Вопрос: {question}
"""

def query_rag(query_text: str):
    db = Chroma(
        collection_name="meds",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )

    print(f"🔍 Документов в базе: {db._collection.count()}")
    print("🔎 Выполняется поиск по векторной базе...")
    results = db.similarity_search_with_score(query_text, k=10)

    print(f"\n📄 Найдено {len(results)} релевантных документов:\n")
    for i, (doc, score) in enumerate(results):
        print(f"🔹 Документ {i+1} | Название: {doc.metadata.get('торговое_название', 'Неизвестно')} | Score: {score:.4f}")
        print(doc.page_content[:300])
        print("------")

    if not results:
        print("❌ Ничего не найдено.")
        return "Ничего не найдено."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results]
    print(f"\n🧠 Ответ:\n{response_text}")
    print(f"\n📚 Источники: {sources}")

    return response_text


def run_rag():
    # Проверка наличия базы
    if os.path.exists(CHROMA_PATH):
        print("📁 Файлы в chroma_db:")
        print(os.listdir(CHROMA_PATH))
    else:
        print(f"❗ Папка {CHROMA_PATH} не найдена!")


    # Проверка количества документов
    vectordb = Chroma(
        collection_name="meds",
        embedding_function=embedding_function(),
        persist_directory=CHROMA_PATH,
    )
    print(f"📊 Документов в базе: {vectordb._collection.count()}")

    # Запрос пользователя
    query_text = input("\n❓ Введите ваш вопрос: ")
    query_rag(query_text)


if __name__ == "__main__":
    run_rag()


# CHROMA_PATH = CHROMA_PATH

# PROMPT_TEMPLATE = """
# Ответьте на следующий вопрос, используя только приведённый ниже контекст:

# {context}

# Вопрос: {question}
# """

# def query_tool(query_text: str) -> str:
#     db = Chroma(
#         collection_name="meds",
#         persist_directory=CHROMA_PATH,
#         embedding_function=embedding_function()
#     )

#     results = db.similarity_search_with_score(query_text, k=10)

#     if not results:
#         return "❌ Ничего не найдено."

#     # Формируем контекст из найденных документов
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # LLM
#     response_text = model.invoke(prompt)

#     # Источники
#     sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results]
#     source_text = ", ".join(sources)

#     return f"🧠 Ответ:\n{response_text}\n\n📚 Источники: {source_text}"


# from llama_index.core.agent.workflow import AgentWorkflow
# import asyncio


# check_agent = AgentWorkflow.from_tools_or_functions(
#         tools_or_functions=[query_tool],
#         llm=model,
#         system_prompt = """
#         Ты — ИИ-агент проверки фактов. 
#         Твоя задача — проверять утверждения пользователя на основе информации из базы данных (через инструмент query_tool).

#         Вот как ты действуешь:
#         1. Получи утверждение от пользователя.
#         2. С помощью инструмента `query_tool` найди релевантную информацию.
#         3. Сравни утверждение с полученным контекстом.
#         4. Скажи, является ли утверждение:
#         - ИСТИННЫМ — если контекст прямо подтверждает его;
#         - ЛОЖНЫМ — если контекст явно ему противоречит;
#         - НЕДОСТАТОЧНО ДАННЫХ — если информации из базы недостаточно для уверенного вывода.

#         Формат ответа:
#         Результат: [ИСТИНА / ЛОЖЬ / НЕДОСТАТОЧНО ДАННЫХ]  
#         Обоснование: [пояснение на основе контекста]

#         Отвечай строго по этому формату.
#         """

#     )

# # Функция проверки утверждения
# async def check_statement(statement: str):
#     """
#     Проверяет утверждение с помощью агента.
#     """
#     result = await check_agent.run(statement)
#     return result

# # Точка входа
# if __name__ == "__main__":
#     statement = input("Введите утверждение: ")
#     response = asyncio.run(check_statement(statement))
#     print(response)