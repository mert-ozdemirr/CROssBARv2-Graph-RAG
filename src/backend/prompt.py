from typing import List

def build_prompt(context_docs: List[str], user_query: str) -> str:
    context = "\n\n".join(context_docs)
    return f"""You are an expert assistant answering questions based on the following context.

Context:
{context}

User Question:
{user_query}

Answer:"""
