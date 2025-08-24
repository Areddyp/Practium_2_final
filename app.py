
import streamlit as st
from article_sources import fetch_articles_newsapi, fetch_articles_googlesearch, get_article_text
from summarizer import (
    extractive_summarizer, hf_summarizer,
    openai_summarize, llama_summarize, perplexity_summarize, t5_finetuned_summarizer
)
from semantic_search import get_keywords, suggest_articles

# UI Header
st.title("Article Summarizer (Choose Your Model)")

# Input Section
query = st.text_input("Enter a topic to search and summarize:")
source_choice = st.radio("Select Article Source:", ["Google Search", "NewsAPI"])
newsapi_key = st.text_input("Enter your NewsAPI key (if using NewsAPI):", type="password") if source_choice == "NewsAPI" else None

summarizer_options = [
    "Extractive (NLTK, frequency-based)",
    "Abstractive (BART, bart-base)",
    "Abstractive (T5, t5-base)",
    "Abstractive (T5, local finetuned t5-small)",
    "Abstractive (Pegasus, pegasus-cnn_dailymail)",
    "OpenAI GPT-3.5/GPT-4 (API key)",
    "Llama/Llama2/Llama3 (public endpoint)",
    "Perplexity (API endpoint)"
]
summarizer_model = st.selectbox("Choose summarizer model:", summarizer_options)

# Options for extractive model
if summarizer_model == "Extractive (NLTK, frequency-based)":
    summarization_length = st.slider("Summarization Length (words)", 50, 500, 150, step=10)
else:
    summarization_length = None

# API inputs
perplexity_api_url = None
if summarizer_model == "Perplexity (API endpoint)":
    perplexity_api_url = st.text_input(
        "Enter your Perplexity AI API endpoint:",
        value="https://api.perplexity.ai/v1/chat/completions"
    )
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password") if "OpenAI" in summarizer_model else None
llama_api_url = st.text_input("Enter your Llama API endpoint:", value="https://api.groq.com/openai/v1/chat/completions") if "Llama" in summarizer_model else None

# Summarize action
if st.button("Search & Summarize") and query:
    with st.spinner("Fetching and summarizing articles..."):
        if source_choice == "NewsAPI":
            if not newsapi_key:
                st.warning("Please provide a NewsAPI key.")
                st.stop()
            articles = fetch_articles_newsapi(newsapi_key, query)
        else:
            articles = fetch_articles_googlesearch(query)

        if not articles:
            st.warning("No articles found.")
            st.stop()

        summary = ''
        article_links = []

        for article in articles:
            article_text = get_article_text(article)
            if not article_text or len(article_text.split()) < 40:
                continue

            if summarizer_model == "Extractive (NLTK, frequency-based)":
                summary_piece = extractive_summarizer(article_text, summarization_length)
            elif summarizer_model == "Abstractive (BART, bart-base)":
                summary_piece = hf_summarizer(article_text, model_name="facebook/bart-base")
            elif summarizer_model == "Abstractive (T5, t5-base)":
                summary_piece = hf_summarizer(article_text, model_name="t5-base")
            elif summarizer_model == "Abstractive (T5, local finetuned t5-small)":
                summary_piece = t5_finetuned_summarizer(article_text, model_name="t5-small")
            elif summarizer_model == "Abstractive (Pegasus, pegasus-cnn_dailymail)":
                summary_piece = hf_summarizer(article_text, model_name="google/pegasus-cnn_dailymail")
            elif summarizer_model == "OpenAI GPT-3.5/GPT-4 (API key)":
                if not openai_api_key:
                    st.warning("OpenAI key is missing.")
                    summary_piece = ""
                else:
                    summary_piece = openai_summarize(article_text, openai_api_key)
            elif summarizer_model == "Llama/Llama2/Llama3 (public endpoint)":
                if not llama_api_url:
                    st.warning("Llama API URL missing.")
                    summary_piece = ""
                else:
                    summary_piece = llama_summarize(article_text, llama_api_url)
            elif summarizer_model == "Perplexity (API endpoint)":
                if not perplexity_api_url:
                    st.warning("Perplexity API URL missing.")
                    summary_piece = ""
                else:
                    summary_piece = perplexity_summarize(article_text, perplexity_api_url)
            else:
                summary_piece = extractive_summarizer(article_text, 150)

            if summary_piece:
                st.markdown(f"#### [{article}]({article})")
                st.write(summary_piece)
                summary += summary_piece + "\n"
                article_links.append(article)

        if summary:
            st.subheader("Keywords from combined summaries:")
            kw = get_keywords(summary)
            st.write(", ".join(kw))

            st.subheader("Suggested Related Articles:")
            suggestions = suggest_articles(articles, summary, get_article_text)
            for s in suggestions:
                st.write(s)
        else:
            st.warning("No summaries could be generated for the found articles.")
