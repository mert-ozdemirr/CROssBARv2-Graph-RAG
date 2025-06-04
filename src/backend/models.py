from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    chat_id: int
    content: str
    model: str
    api_key: str
    searchLength: int
    extensionSize: int

class QueryResponse(BaseModel):
    chat_id: int
    response: str
