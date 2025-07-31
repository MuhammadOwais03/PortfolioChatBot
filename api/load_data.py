from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain_core.embeddings import Embeddings
from langchain_cohere import CohereEmbeddings


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
    # embeddings = HuggingFaceEmbeddings(model_name=model_name)
    model = INSTRUCTOR(model_name)
    return model


class CustomInstructorEmbedding(Embeddings):
    def __init__(self, model_name="hkunlp/instructor-xl", instruction="Represent the sentence for semantic search:"):
        self.model = INSTRUCTOR(model_name)
        self.instruction = instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode([[self.instruction, text] for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([[self.instruction, text]])[0]
    
    
def download_cohere_embedding(API_KEY):
    embeddings = CohereEmbeddings(cohere_api_key=API_KEY, model="embed-multilingual-v2.0")
    return embeddings