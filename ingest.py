from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def build_vector_databse():
    loader = PyPDFDirectoryLoader('./data')
    documents = loader.load()

    print(f"DEBUG: Found {len(documents)} pages in the data folder.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory='./chroma_db'
    )


if __name__ == '__main__':
    build_vector_databse()