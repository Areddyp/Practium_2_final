
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from heapq import nlargest
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def get_keywords(text, num_keywords=5):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = Counter(filtered_words)
    return nlargest(num_keywords, word_freq, key=word_freq.get)

def suggest_articles(articles, current_text, get_article_text_func, num_suggestions=3):
    current_keywords = set(get_keywords(current_text))
    suggested_articles = []
    for article in articles:
        article_text = get_article_text_func(article)
        if not article_text:
            continue
        article_keywords = set(get_keywords(article_text))
        similarity = len(current_keywords.intersection(article_keywords))
        suggested_articles.append((article, similarity))
    suggested_articles.sort(key=lambda x: x[1], reverse=True)
    return [art[0] for art in suggested_articles[:num_suggestions]]
