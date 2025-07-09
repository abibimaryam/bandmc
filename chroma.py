import os
import shutil
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from embedding_functiom import embedding_function
from config import DATA_DIR,CHROMA_PATH


# Пути
DATA_DIR = DATA_DIR
CHROMA_PATH = CHROMA_PATH

# Инициализируем эмбеддинг-функцию один раз
embedding_func = embedding_function()


def load_all_documents(data_dir: str) -> list[Document]:
    """Считывает все Excel-файлы в папке и формирует список Document."""
    excel_files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx")]
    docs = []

    for file in excel_files:
        path = os.path.join(data_dir, file)
        df = pd.read_excel(path, skiprows=1)
        df = df.fillna("")
        df.columns = [' '.join(col.strip().split()) for col in df.columns]

        for idx, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            try:
                metadata = {
                    "торговое_название": row.get("Торговое название и синоним", ""),
                    "международное_название": row.get("Международное название", ""),
                    "лекарственная_форма": row.get("Лекарственная форма выпуска", ""),
                    "страна_производитель": row.get("Страна-производитель", ""),
                    "фирма_производитель": row.get("Фирма производитель", ""),
                    "фармакотерапевтическая_группа": row.get("Фармакотерапевтическая группа", ""),
                    "файл": file
                }

                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
            except Exception as e:
                print(f"⚠️ Ошибка в строке {idx} файла {file}: {e}")
    return docs


def reset_chroma():
    """Удаляет старую базу и пересоздаёт её с новыми данными."""
    if os.path.exists(CHROMA_PATH):
        print("♻️ Удаление старой базы Chroma...")
        shutil.rmtree(CHROMA_PATH)

    print("📥 Загрузка документов...")
    docs = load_all_documents(DATA_DIR)
    print(f"📄 Всего документов: {len(docs)}")

    vectordb = Chroma(
        collection_name="meds",
        embedding_function=embedding_func,
        persist_directory=CHROMA_PATH,
    )

    vectordb.add_documents(docs)
    vectordb.persist()
    print(f"✅ Загружено {len(docs)} документов в новую ChromaDB.")
    print(f"📦 Документов в базе после сброса: {vectordb._collection.count()}")


def update_chroma():
    """Добавляет новые документы в уже существующую ChromaDB."""
    print("📥 Загрузка документов...")
    docs = load_all_documents(DATA_DIR)
    print(f"📄 Всего документов: {len(docs)}")

    vectordb = Chroma(
        collection_name="meds",
        embedding_function=embedding_func,
        persist_directory=CHROMA_PATH,
    )

    vectordb.add_documents(docs)
    vectordb.persist()
    print(f"✅ Добавлено {len(docs)} документов в ChromaDB.")
    print(f"📦 Документов в базе после добавления: {vectordb._collection.count()}")


# Пример использования:
# reset_chroma()  # Для полного сброса и загрузки заново
# update_chroma()  # Чтобы просто добавить документы
