from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI / Google Gemini client
client = OpenAI(
    api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
)

# Qdrant Cloud configuration
QDRANT_CLOUD_URL = os.getenv("QDRANT_LINK") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")     

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY,
    collection_name="about_ayush",
        timeout=60  # seconds

)



def get_chat_response(user_query: str):
    search_result = vector_db.similarity_search(query=user_query)
    context = "\n".join([doc.page_content for doc in search_result])

    SYSTEM_PROMPT = f"""
    You are Ayush Gupta’s personal portfolio assistant.
    Use ONLY the information below to answer the user query.
    
    If something is not in the context just say sorry I have no idea about it .
    and if someone ask vulgular ques or sexual just reply him aggressively like below but keep remember i should be present always as nice person
    
    If the reponse has a kind of list Represent its as a numbered list.
 

    information:
    {context}
    """
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
    )

    return response.choices[0].message.content
