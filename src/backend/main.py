from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

from models import QueryRequest, QueryResponse
from prompt import build_prompt
from llm_router import generate_response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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
        prompt = build_prompt(data.retrieved_docs, data.content)
        answer = generate_response(model=data.model, prompt=prompt, api_key=data.api_key)
        return QueryResponse(chat_id=data.chat_id, response=answer)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
