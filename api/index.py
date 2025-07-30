from http.server import BaseHTTPRequestHandler
import json
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from load_data import download_hugging_face_embeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "portfolio-chatbot"

llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.5-pro")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit(1)
    
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)  

# Convert your existing prompt template to chat format
system_template = """You are an AI assistant for Muhammad Owais's portfolio.

Context:
{context}

Answer accordingly using only the context provided. Be clear, and helpful."""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

chat_prompt = ChatPromptTemplate.from_messages(messages)

# Initialize memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Use ConversationalRetrievalChain with your custom chat prompt
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    return_source_documents=True,
    verbose=True
)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        message = {"message": "Hello from Vercel's Python server!"}
        self.wfile.write(json.dumps(message).encode())
        
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode("utf-8"))
            query = body.get("query", "")
            
            if not query:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Query not provided"}).encode())
                return

            response = qa.invoke({"question": query})
            
            answer = response.get("answer", "No answer found.")
            sources = [doc.metadata for doc in response.get("source_documents", [])]

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            result = {
                "answer": answer,
                "sources": sources
            }

            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

if __name__ == "__main__":
    from http.server import HTTPServer
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, handler)
    print("Server running on http://localhost:8000")
    httpd.serve_forever()