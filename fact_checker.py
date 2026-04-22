import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from dotenv import load_dotenv
from ner_utils import extract_entities    # Import NER

# Load .env
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("GNEWS_API_KEY")


# Build Smart Query using NER
def build_query(text):
    try:
        entities = extract_entities(text)

        entity_words = [ent[0] for ent in entities]

        if entity_words:
            query = " ".join(entity_words[:5])
        else:
            query = text

        # CLEAN QUERY
        query = query.replace(".", "")
        query = re.sub(r'[^a-zA-Z0-9 ]', '', query)

        stopwords = {"the", "is", "in", "at", "of", "on", "and", "a", "an"}
        words = [w for w in query.split() if w.lower() not in stopwords]

        query = " ".join(words[:5])

        return query

    except Exception as e:
        print("QUERY BUILD ERROR:", e)
        return " ".join(text.split()[:5])


# Fetch News
def fetch_news(query):
    try:
        query = build_query(query)
        print("SMART QUERY:", query)

        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=10&token={API_KEY}"
        print("API URL:", url)

        response = requests.get(url)
        data = response.json()

        # Handle API errors
        if "errors" in data:
            print("API ERROR:", data["errors"])

            if "too many requests" in str(data["errors"]).lower():
                return "rate_limited"

            return None

        articles = data.get("articles", [])
        print("ARTICLES FOUND:", len(articles))

        news_list = []

        for article in articles:
            text = article.get("title", "") + " " + str(article.get("description", ""))
            news_list.append(text)

        return news_list

    except Exception as e:
        print("FETCH ERROR:", e)
        return None


# Similarity
def get_similarity(input_text, news_list):
    if not news_list:
        return None

    try:
        corpus = [input_text] + news_list

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        similarity = cosine_similarity(vectors[0:1], vectors[1:])

        return max(similarity[0])

    except Exception as e:
        print("SIMILARITY ERROR:", e)
        return None


#  Fact Check
def fact_check(input_text):
    news_list = fetch_news(input_text)

    # Handle rate limit
    if news_list == "rate_limited":
        return {"status": "rate_limited"}

    if news_list is None:
        return {"status": "api_failed"}

    # No fallback (as per fix)
    if len(news_list) == 0:
        return {"status": "no_data"}

    similarity = get_similarity(input_text, news_list)

    return {
        "status": "success",
        "similarity": round(float(similarity), 2) if similarity else 0,
        "matched_news": news_list[:3]
    }