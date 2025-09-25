import wikipediaapi
from newsapi import NewsApiClient

# Wikipedia retriever
def retrieve_wikipedia(query, lang="en", max_results=3):
    wiki_wiki = wikipediaapi.Wikipedia(lang)
    page = wiki_wiki.page(query)
    if page.exists():
        # split text into chunks of ~500 chars
        text = page.text
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        return chunks[:max_results]
    return []

# News retriever (live news)
def retrieve_news(query, api_key, max_results=3):
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language="en", page_size=max_results)
    chunks = []
    for article in articles['articles']:
        title = article.get('title', '')
        desc = article.get('description', '')
        content = article.get('content', '')
        combined = " ".join([title, desc, content])
        chunks.append(combined)
    return chunks
