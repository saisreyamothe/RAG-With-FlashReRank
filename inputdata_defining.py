from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
def run_ingestion():
    loader=TextLoader("Data/Animal_info.txt")
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Vector_DB=Chroma.from_documents(
        documents=chunks,
        embeddings=embeddings,
        persist_directory="Vector_DB"
    )
    print("Success! Your vector database has been created in the Vector_DB directory.")
if __name__=="__main__":
    run_ingestion()


