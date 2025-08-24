
import requests
from bs4 import BeautifulSoup

def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except Exception as e:
        return f"Error fetching article: {str(e)}"

def fetch_articles_newsapi(api_key, query, page_size=5):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])
        return [article["url"] for article in articles if "url" in article]
    except Exception as e:
        return []

def fetch_articles_googlesearch(query, num_results=5):
    try:
        from googlesearch import search as google_search
        return list(google_search(query, num_results=num_results))
    except Exception as e:
        return []
