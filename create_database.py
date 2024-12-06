import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from get_embeddings import get_embeddings
from tqdm import tqdm  

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    try:
        documents = load_documents_one_by_one()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    except Exception as e:
        print(f"❌ Error: {e}")


def load_documents_one_by_one():
    documents = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file_name)
            try:
                print(f"📄 Attempting to load: {file_name}")
                loader = PyPDFLoader(file_path)  # Initialize the loader for the specific file
                pdf_documents = loader.load()   # Load the document
                documents.extend(pdf_documents)
                print(f"✅ Successfully loaded {len(pdf_documents)} pages from {file_name}.")
            except Exception as e:
                print(f"⚠️ Error loading {file_name}: {e}")
    return documents


def split_documents(documents: list[Document]):
    if not documents:
        print("⚠️ No documents to split.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    print(f"🔄 Splitting {len(documents)} documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks.")
    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embeddings()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        
        # Use tqdm to show progress
        for i in tqdm(range(len(new_chunks)), desc="Adding Chunks"):
            db.add_documents([new_chunks[i]], ids=[new_chunk_ids[i]])  # Add one document at a time
        
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "unknown")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

        print(f"🔢 Assigned chunk ID: {chunk_id}")

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()