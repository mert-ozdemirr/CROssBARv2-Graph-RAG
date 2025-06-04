from typing import List

def build_prompt(context_docs: List[str], user_query: str) -> str:
    context = "\n\n".join(context_docs)
    return f"""
User Question:
{user_query}    

Context:
{context}
"""
