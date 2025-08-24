
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from heapq import nlargest
from transformers import pipeline
import requests
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import re
import torch
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

def data_cleaning(row):
        
    #ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!
    
    row=re.sub("(\\t)", ' ', str(row)).lower() #remove escape charecters
    row=re.sub("(\\r)", ' ', str(row)).lower() 
    row=re.sub("(\\n)", ' ', str(row)).lower()
    
    row=re.sub("(__+)", ' ', str(row)).lower()   #remove _ if it occors more than one time consecutively
    row=re.sub("(--+)", ' ', str(row)).lower()   #remove - if it occors more than one time consecutively
    row=re.sub("(~~+)", ' ', str(row)).lower()   #remove ~ if it occors more than one time consecutively
    row=re.sub("(\+\++)", ' ', str(row)).lower()   #remove + if it occors more than one time consecutively
    row=re.sub("(\.\.+)", ' ', str(row)).lower()   #remove . if it occors more than one time consecutively
    
    #row=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower() #remove <>()|&©ø"',;?~*!
    row=re.sub(r"[<>()|&©ø\[\]\'\;~*]", ' ', str(row)).lower() #remove <>()|&©ø"',;?~*!
    
    row=re.sub("(mailto:)", ' ', str(row)).lower() #remove mailto:
    row=re.sub(r"(\\x9\d)", ' ', str(row)).lower() #remove \x9* in text
    row=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower() #replace INC nums to INC_NUM
    row=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower() #replace CM# and CHG# to CM_NUM
    
    
    row=re.sub("(\.\s+)", ' ', str(row)).lower() #remove full stop at end of words(not between)
    row=re.sub("(\-\s+)", ' ', str(row)).lower() #remove - at end of words(not between)
    row=re.sub("(\:\s+)", ' ', str(row)).lower() #remove : at end of words(not between)
    
    row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
    
    #Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
        repl_url = url.group(3)
        row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(row))
    except:
        pass #there might be emails with no url in them
    
    row = re.sub("(\s+)",' ',str(row)).lower() #remove multiple spaces
    
    #Should always be last
    row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces

    tokens = word_tokenize(row)
    lemmatized_row = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    
    return lemmatized_row

def extractive_summarizer(text, target_length=150):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    word_freq = {}
    for word in words:
        if word.isalnum() and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
    summary_sentences = nlargest(max(1, min(len(sentences), target_length // 20)), sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def get_hf_summarizer(model_name):
    return pipeline("summarization", model=model_name)

def hf_summarizer(text, model_name):
    summarizer = get_hf_summarizer(model_name)
    try:
        return summarizer(text, max_length=180, min_length=30, do_sample=False, truncation=True)[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

def t5_finetuned_summarizer(text, model_name):
    text = data_cleaning(text)
    tokenizer = AutoTokenizer.from_pretrained("../working/tokenizer", local_file_only=True)
    pipe = pipeline("summarization", model="../t5-news/checkpoint-11000", tokenizer=tokenizer, device=device)
    try:
        return pipe([text])[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

def openai_summarize(article_text, api_key, model_version="gpt-3.5-turbo"):
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(model=model_version,
        messages=[{"role": "user", "content": f"Summarize this article:\n\n{article_text}"}],
        temperature=0.2,
        max_tokens=600)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Summarization Error: {str(e)}"

def llama_summarize(article_text, api_url):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "messages": [
            {"role": "user", "content": f"Summarize this article:\n{article_text}"}
        ]
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json().choices[0].message.content.strip()
    except Exception as e:
        return f"Llama Summarization Error: {str(e)}"

def perplexity_summarize(article_text, api_url):
    """
    Send a summary request to Perplexity API following OpenAI-compatible payload.
    """
    import requests
    headers = {'Content-Type': 'application/json'}
    payload = {
        "messages": [
            {"role": "user", "content": f"Summarize this article:\n{article_text}"}
        ]
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        # Perplexity and other OpenAI-compatible APIs typically return:
        # {'choices': [{'message': {'content': ...}}]}
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Perplexity Summarization Error: {str(e)}"

