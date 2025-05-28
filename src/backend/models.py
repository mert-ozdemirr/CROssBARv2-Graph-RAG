from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    chat_id: str
    content: str
    model: str
    api_key: str
    searchLength: int
    extensionSize: int
    retrieved_docs: List[str]

class QueryResponse(BaseModel):
    chat_id: str
    response: str
