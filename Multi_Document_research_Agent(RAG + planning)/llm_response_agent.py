from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API_KEY
)

def generate_response(query, top_chunks):
    context = "\n".join([chunk for chunk, _ in top_chunks])
    prompt = f"""
You are an AI research assistant for Retrieval-Augmented Generation (RAG) with planning capabilities.
Answer the question using ONLY the provided context. Include traceable sources.

Context:
{context}

Question: {query}

Instructions:
1. Provide a concise answer.
2. Generate 3-5 bullet points summarizing key info.
3. List sources for each bullet point.
4. Respond in JSON format with keys: question, answer, bullet_points, sources.

Answer:
"""
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {str(e)}")
        return {
            "question": query,
            "answer": "Error generating response",
            "bullet_points": [],
            "sources": []
        }

    # Provide sources from retrieved chunks
    sources = [doc_name for _, doc_name in top_chunks]
    return answer, sources
