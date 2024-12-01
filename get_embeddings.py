from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
