from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_storage"
))

# Create collections for chats and messages if they don't exist
try:
    chats_collection = chroma_client.get_or_create_collection(name="chats")
    messages_collection = chroma_client.get_or_create_collection(name="messages")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")

class Query(BaseModel):
    content: str
    model: str
    chat_id: int

class Chat(BaseModel):
    name: str

@app.post("/query")
async def process_query(query: Query):
    try:
        # Store the message
        message_id = f"msg_{datetime.now().timestamp()}"
        messages_collection.add(
            documents=[json.dumps({
                "content": query.content,
                "role": "user",
                "timestamp": datetime.now().isoformat(),
                "chat_id": query.chat_id
            })],
            ids=[message_id]
        )

        # For now, return a simple response
        response_content = f"Received query: {query.content} using model: {query.model}"
        
        # Store the response
        response_id = f"msg_{datetime.now().timestamp()}_response"
        messages_collection.add(
            documents=[json.dumps({
                "content": response_content,
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "chat_id": query.chat_id
            })],
            ids=[response_id]
        )

        return {
            "response": response_content,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats")
async def create_chat(chat: Chat):
    try:
        chat_id = f"chat_{datetime.now().timestamp()}"
        chats_collection.add(
            documents=[json.dumps({
                "name": chat.name,
                "created_at": datetime.now().isoformat()
            })],
            ids=[chat_id]
        )
        return {"id": chat_id, "name": chat.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str):
    try:
        results = messages_collection.get(
            where={"chat_id": chat_id}
        )
        messages = [json.loads(doc) for doc in results["documents"]]
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)