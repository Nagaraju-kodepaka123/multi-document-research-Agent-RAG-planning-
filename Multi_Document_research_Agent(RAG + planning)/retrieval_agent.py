from mcp import send_mcp_message
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import wikipedia

# Initialize embedder and FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []
document_names = []

# NewsAPI key
NEWS_API_KEY = "808f9eb4acdf498c94c6c3adb5de73eb"

def store_embeddings(chunks_in, document_name):
    """Vectorize and store document chunks in FAISS"""
    global index, chunks, document_names
    embeddings = embedder.encode(chunks_in, convert_to_numpy=True)
    if index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    chunks.extend(chunks_in)
    document_names.extend([document_name] * len(chunks_in))

def fetch_news(query, top_k=3):
    """Fetch top-k news articles from NewsAPI"""
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize={top_k}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    articles = response.json().get("articles", [])
    news_chunks = []
    for article in articles:
        content = article.get("title", "") + " " + article.get("description", "")
        news_chunks.append((f"{content}", f"NewsAPI: {article.get('url')}"))
    return news_chunks

def fetch_wikipedia(query, sentences=3):
    """Fetch Wikipedia summary"""
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return [(summary, "Wikipedia")]
    except Exception:
        return []

def retrieve_chunks(query, k_local=3, k_wiki=2, k_news=2):
    """Planning agent: retrieve from multiple sources and vectorize online sources"""
    global index, chunks, document_names
    top_chunks = []

    # --- Local docs retrieval ---
    if index is not None and len(chunks) > 0:
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k_local)
        top_chunks.extend([(chunks[i], document_names[i]) for i in indices[0]])

    # --- Wikipedia retrieval ---
    wiki = fetch_wikipedia(query, sentences=k_wiki)
    if wiki:
        wiki_texts, wiki_sources = zip(*wiki)
        # Vectorize Wikipedia chunks for unified retrieval
        store_embeddings(list(wiki_texts), "Wikipedia")
        top_chunks.extend(wiki)

    # --- NewsAPI retrieval ---
    news = fetch_news(query, top_k=k_news)
    if news:
        news_texts, news_sources = zip(*news)
        store_embeddings(list(news_texts), "NewsAPI")
        top_chunks.extend(news)

    # Send to MCP
    send_mcp_message(
        "RetrievalAgent", "LLMResponseAgent", "CONTEXT_RESPONSE",
        {"top_chunks": top_chunks, "query": query}
    )

    return top_chunks
