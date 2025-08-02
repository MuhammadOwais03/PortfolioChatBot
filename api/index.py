from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from load_data import download_cohere_embedding

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = "portfolio-chatbot-1"

llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.5-pro"
)

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")

embeddings = download_cohere_embedding(COHERE_API_KEY)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

system_template = """You are an AI assistant for Muhammad Owais's portfolio.

Context:
{context}

Answer accordingly using only the context provided. Be clear, and helpful."""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

chat_prompt = ChatPromptTemplate.from_messages(messages)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
    return_source_documents=True,
    verbose=True
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json()
        query = body.get("query", "")

        if not query:
            return jsonify({"error": "Query not provided"}), 400

        response = qa.invoke({"question": query})
        answer = response.get("answer", "No answer found.")
        sources = [doc.metadata for doc in response.get("source_documents", [])]

        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WSGI entry point for Vercel
def handler(environ, start_response):
    return app(environ, start_response)
