import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re

# Check if input is URL
def is_url(text):
    return text.startswith("http://") or text.startswith("https://")

# Extract text from URL
from newspaper import Article

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        return article.text

    except Exception as e:
        print("Extraction Error:", e)
        return ""
    

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Trust Score
def get_trust_score(url):
    trusted = ["bbc.com", "reuters.com", "thehindu.com", "ndtv.com"]

    for site in trusted:
        if site in url:
            return "High ✅"
    
    return "Unknown ⚠️"