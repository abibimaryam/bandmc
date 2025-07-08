import argparse
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_functiom import embedding_function  # Проверьте правильность имени модуля!

CHROMA_PATH = "chroma_db"

# PROMPT_TEMPLATE = """
# Ответьте на следующий вопрос, используя только приведённый ниже контекст:

# {context}

# Вопрос: {question}
# """

PROMPT_TEMPLATE = """
Выступите в роли системы проверки утверждений на основе фактов.

На основе приведённого контекста определите, является ли утверждение ИСТИННЫМ, ЛОЖНЫМ или НЕИЗВЕСТНЫМ (если в контексте нет достаточной информации). Обязательно объясните ваш вывод, ссылаясь на факты из контекста.

Контекст:
{context}

Утверждение: {question}

Формат ответа:
- Метка: ИСТИННО / ЛОЖНО / НЕИЗВЕСТНО
- Обоснование: краткое объяснение на основе контекста

Ответ:
"""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Вопрос для RAG.")
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    # Загружаем векторную БД с указанием collection_name
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

    # Формируем промпт для модели
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Инициализируем модель Ollama
    model = Ollama(model="gemma3:4b",temperature=0)
    response_text = model.invoke(prompt)

    # Источники для ответа
    sources = [doc.metadata.get("торговое_название", "неизвестно") for doc, _ in results]
    print(f"\n🧠 Ответ:\n{response_text}")
    print(f"\n📚 Источники: {sources}")

    return response_text

if __name__ == "__main__":
    # Выводим файлы базы для проверки
    if os.path.exists(CHROMA_PATH):
        print("Файлы в chroma_db после сохранения:")
        print(os.listdir(CHROMA_PATH))
    else:
        print(f"Папка {CHROMA_PATH} не найдена!")

    # Проверяем размер и пример эмбеддинга
    emb = embedding_function()
    vec = emb.embed_query("ампициллин натриевая соль")
    print(f"Размер эмбеддинга: {len(vec)}")
    print(f"Пример эмбеддинга (первые 5 значений): {vec[:5]}")

    # Проверяем количество документов в базе
    vectordb = Chroma(
        collection_name="meds",
        embedding_function=embedding_function(),
        persist_directory=CHROMA_PATH,
    )
    print(f"Документов в базе после добавления: {vectordb._collection.count()}")

    # Запускаем основной обработчик
    main()
