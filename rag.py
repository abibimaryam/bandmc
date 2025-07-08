import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_functiom import embedding_function
from config import CHROMA_PATH,model



CHROMA_PATH = CHROMA_PATH

PROMPT_TEMPLATE = """
Вы — система проверки утверждений на основе контекста.

Проанализируйте утверждение и найдите ВСЕ несоответствия с приведённым ниже контекстом. 
Укажите, какие именно части утверждения противоречат фактам, приведённым в контексте. 
Если утверждение полностью соответствует контексту — просто скажите: "Несоответствий не найдено."

Формат ответа:
[список несоответствий или "Несоответствий не найдено."]

Контекст:
{context}

Утверждение: {question}
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
    query_text = input("\n❓ Введите ваше утверждение: ")
    query_rag(query_text)


if __name__ == "__main__":
    run_rag()

