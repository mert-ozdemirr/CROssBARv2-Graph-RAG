from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import pickle
import google.generativeai as genai


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
def handle_query(request: Request, data: QueryRequest):
    #try:
    logger.info(f"Received query: {data.content} using model: {data.model}")
    print(data.content)
    retrieved_docs = graph_retriever(data.content, data.searchLength, data.extensionSize, request)
    prompt = build_prompt(retrieved_docs, data.content)
    print(len(prompt))
    print(prompt[0])
    print(prompt[-1])


    client = genai.configure(api_key='AIzaSyDbMReeHt_IAjjLqDe2OVTJPFPClJLshBQ')
    model=genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-04-17",
    system_instruction="Please use the structured data I provide to you as knowledge base and answer the question in the prompt accordingly. And do not use any additional information you know externally.")
    
    if len(prompt) > 3000000:
        prompt = prompt[:3000000]

    response = model.generate_content(prompt)
    answer = response.text
    
    #answer = generate_response(model=data.model, prompt=prompt, api_key=data.api_key)
    return QueryResponse(chat_id=data.chat_id, response=answer)
    """except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))"""
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

