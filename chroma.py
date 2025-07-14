import os
import shutil
import pandas as pd
from langchain_community.vectorstores import Chroma  # ← Если обновилась: замени на `from langchain_chroma import Chroma`
from langchain.schema import Document
from embedding_functiom import embedding_function
from config import DATA_DIR, CHROMA_PATH


# Инициализация эмбеддинга
embedding_func = embedding_function()


def parse_lekarstva(df: pd.DataFrame, file: str) -> list[Document]:
    docs = []
    for idx, row in df.iterrows():
        try:
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            metadata = {
                "тип": "медикаменты",
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
            print(f"⚠️ Ошибка обработки строки {idx} в {file}: {e}")
    return docs

def parse_bads(df: pd.DataFrame, file: str) -> list[Document]:
    docs = []
    for idx, row in df.iterrows():
        try:
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            metadata = {
                "тип": "бад",
                "наименование": row.get("Наименование БАДов", ""),
                "организация": row.get("Название организации", ""),
                "файл": file
            }
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        except Exception as e:
            print(f"⚠️ Ошибка обработки строки {idx} в {file}: {e}")
    return docs


def load_all_documents(data_dir: str) -> list[Document]:
    excel_files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx")]
    docs = []

    for file in excel_files:
        path = os.path.join(data_dir, file)

        try:
            # Определение нужно ли пропускать первую строку
            skip = 1 if "лек" in file.lower() else 0
            df = pd.read_excel(path, skiprows=skip)
        except Exception as e:
            print(f"❌ Ошибка чтения файла {file}: {e}")
            continue

        df = df.fillna("")
        df.columns = [' '.join(str(col).strip().split()) for col in df.columns]

        # Определение типа по названию файла
        if "лек" in file.lower():
            parsed_docs = parse_lekarstva(df, file)
        elif "бад" in file.lower():
            parsed_docs = parse_bads(df, file)
        else:
            print(f"⚠️ Неизвестный тип файла: {file}")
            parsed_docs = []

        docs.extend(parsed_docs)

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

    batch_size = 5000
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectordb.add_documents(batch)
        print(f"✅ Загружено: {i + len(batch)} / {len(docs)}")

    vectordb.persist()
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

    batch_size = 5000
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectordb.add_documents(batch)
        print(f"✅ Добавлено: {i + len(batch)} / {len(docs)}")

    vectordb.persist()
    print(f"📦 Документов в базе после добавления: {vectordb._collection.count()}")


# Пример использования:
if __name__ == "__main__":
    # reset_chroma()  # Для полного сброса и загрузки заново
    # update_chroma()  # Чтобы просто добавить документы
    pass