import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from load_data import load_pdf, text_splitter, download_hugging_face_embeddings
import time

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "portfolio-chatbot"

print(f"Using Pinecone API Key: {PINECONE_API_KEY[:10]}..." if PINECONE_API_KEY else "No API key found")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit(1)

try:
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    print(f"Existing indexes: {existing_indexes}")
    
    if INDEX_NAME in existing_indexes:
        print(f"Index {INDEX_NAME} already exists")
    else:
        print(f"Creating serverless index: {INDEX_NAME}")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            print("Waiting for index to be ready...")
            while True:
                try:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.status['ready']:
                        print("Index is ready!")
                        break
                    time.sleep(2)
                except Exception as e:
                    print(f"Waiting... {e}")
                    time.sleep(2)
                    
        except Exception as e:
            print(f"Error creating index: {e}")
            if "already exists" in str(e).lower():
                print("Index already exists, continuing...")
            else:
                exit(1)
        
except Exception as e:
    print(f"Error with index operations: {e}")
    print("Continuing anyway...")

print("Loading PDF data...")
try:
    extracted_data = load_pdf("Data")
    print(f"Loaded {len(extracted_data)} documents")
    
    if not extracted_data:
        print("No documents loaded. Check your Data folder and PDF files.")
        exit(1)
        
except Exception as e:
    print(f"Error loading PDF: {e}")
    exit(1)

print("Splitting text into chunks...")
try:
    text_chunks = text_splitter(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    
    if not text_chunks:
        print("No text chunks created.")
        exit(1)
        
except Exception as e:
    print(f"Error splitting text: {e}")
    exit(1)

print("Loading embeddings model...")
try:
    embeddings = download_hugging_face_embeddings()
    
    test_embedding = embeddings.embed_query("test")
    print(f"Embedding dimension: {len(test_embedding)}")
    
except Exception as e:
    print(f"Error loading embeddings: {e}")
    exit(1)

print("Storing embeddings in Pinecone...")
try:
    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata if hasattr(doc, 'metadata') else {} for doc in text_chunks]
    
    print(f"Processing {len(texts)} texts...")
    
    docsearch = LangchainPinecone.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=INDEX_NAME
    )
    print("Successfully stored embeddings in Pinecone!")
    
    print("Testing vector store...")
    results = docsearch.similarity_search("medical", k=1)
    print(f"Test query returned {len(results)} results")
    if results:
        print(f"Sample result: {results[0].page_content[:100]}...")
    
except Exception as e:
    print(f"Error storing embeddings: {e}")
    
    print("\nTrying modern LangChain-Pinecone integration...")
    try:
        from langchain_pinecone import PineconeVectorStore
        
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        print("Successfully stored embeddings using modern approach!")
        
        results = docsearch.similarity_search("medical", k=1)
        print(f"Test query returned {len(results)} results")
        
    except Exception as e2:
        print(f"Modern approach also failed: {e2}")
        print("\nTroubleshooting steps:")
        print("1. Check Pinecone dashboard for your index")
        print("2. Verify API key permissions")
        print("3. Try creating index manually in dashboard")