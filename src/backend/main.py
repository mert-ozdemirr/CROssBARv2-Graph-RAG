from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import pickle

import retriever
import prompt
import llm_router
from boot import system_boot_bm25
from retriever import graph_retriever
from models import QueryRequest, QueryResponse
from prompt import build_prompt
from llm_router import generate_response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.all_bm25s = system_boot_bm25()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse)
def handle_query(data: QueryRequest):
    try:
        logger.info(f"Received query: {data.content} using model: {data.model}")
        print(data.content)
        retrieved_docs = graph_retriever(data.content, data.searchLength, data.extensionSize)
        prompt = build_prompt(retrieved_docs, data.content)
        #answer = generate_response(model=data.model, prompt=prompt, api_key=data.api_key)
        return QueryResponse(chat_id=data.chat_id, response="answer")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

