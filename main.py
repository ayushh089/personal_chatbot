from fastapi import FastAPI
from pydantic import BaseModel
from chat import get_chat_response
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://guptaayushportfolio.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class QueryRequest(BaseModel):
    query: str


# API endpoint
@app.post("/chat")
async def chat(request: QueryRequest):
    user_query = request.query
    print(user_query)
    response = get_chat_response(user_query)
    return {"response": response}
