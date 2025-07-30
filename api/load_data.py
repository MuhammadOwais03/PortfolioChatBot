from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents


def text_splitter(extracted_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    print(text_splitter, type(extracted_text))
    page = [page.page_content for page in extracted_text]
    text_chunks = text_splitter.create_documents(page)
    print(f"Number of text chunks created: {len(text_chunks)}")
    
    return text_chunks


def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings