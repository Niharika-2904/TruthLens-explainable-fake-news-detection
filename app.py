from flask import Flask, render_template, request, redirect, session, url_for, send_file, flash
import requests
from utils import is_url, extract_text_from_url, clean_text, get_trust_score
from auth import save_prediction
from auth import get_user_history
from shap_explain import get_shap_explanation
import joblib
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explain import get_explanation
from auth import init_db, register_user, login_user, create_admin, DB_PATH
from fact_checker import fact_check
from ner_utils import extract_entities

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from dotenv import load_dotenv
import os
import time
import io
import re

load_dotenv()   # loads variables from .env

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
lr = joblib.load(os.path.join(BASE_DIR, "models/logistic_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl"))

init_db()
create_admin()

# Function to generate PDF report for admin and users using ReportLab library
def generate_pdf(data, title):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"<b>{title}</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    for i, row in enumerate(data, start=1):
        if len(row) == 4:
            # USER
            news = str(row[1])
            prediction = str(row[2])
            confidence = str(row[3])

        elif len(row) == 5: 
            # ADMIN
            news = str(row[2])
            prediction = str(row[3])
            confidence = str(row[4])

        # CLEAN SYMBOLS
        prediction = prediction.replace("■", "").strip()
        confidence = confidence.replace("■", "").strip()

        # OPTIONAL ICONS
        if "REAL" in prediction:
            prediction = "REAL NEWS ✔"
        elif "FAKE" in prediction:
            prediction = "FAKE NEWS ✖"

        elements.append(Paragraph(f"<b>{i}. News:</b> {news}", styles['Normal']))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Normal']))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles['Normal']))
        elements.append(Spacer(1, 15))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Function for the password validation
def is_strong_password(password):
    # Rule:
    # at least 8 characters
    # 1 uppercase
    # 1 lowercase
    # 1 digit
    # 1 special character

    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$'
    return re.match(pattern, password)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Only validation here
        if not is_strong_password(password):
            flash("Password must contain uppercase, lowercase, number, special character and be 8+ characters long", "danger")
            return redirect(url_for('register'))

        # Pass RAW password (not hashed)
        if register_user(username, password):
            flash("Registration Successful!", "success")
            return redirect("/login")
        else:
            return render_template("register.html", error="User already exists")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        role = login_user(username, password)

        if role:
            session["user"] = username
            session["role"] = role

            if role == "admin":
                return redirect(url_for("admin"))
            else:
                return redirect(url_for("predict"))

        else:
            error = "Invalid credentials" 

    return render_template("login.html", error=error)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    user_input = ""

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    final_prediction = None
    explanation = []
    shap_explanation = []
    confidence = None
    trust = None

    #  NEW VARIABLES
    fact_result = None
    entities = []

    if request.method == "POST":
        user_input = request.form["news"]

        if not user_input:
            return render_template("predict.html", error="Please enter text or URL ❌")

        # HANDLE URL OR TEXT
        if is_url(user_input):
            text = extract_text_from_url(user_input)
            
            print("EXTRACTED TEXT:", text[:200])

            if not text:
                return render_template("predict.html", error="Could not extract text from URL ❌")

            trust = get_trust_score(user_input)
        else:
            text = user_input

        # CLEAN TEXT
        cleaned = clean_text(text)
        cleaned = cleaned[:1000]

        # MODEL
        vector = tfidf.transform([cleaned])
        result = lr.predict(vector)[0]
        proba = lr.predict_proba(vector)[0]
        confidence = round(max(proba) * 100, 2)

        if result == 0:
            prediction = "FAKE NEWS ❌"
        else:
            prediction = "REAL NEWS ✅"

        #  NER (Entity Extraction)
        try:
            entities = extract_entities(text)
        except Exception as e:
            print("NER Error:", e)
            entities = []

        # FACT CHECK
        try:
            fact_result = fact_check(text)
        except Exception as e:
            print("Fact Check Error:", e)
            fact_result = {"status": "api_failed"}

        # HYBRID LOGIC (ML + FACT CHECK)
        if fact_result and fact_result.get("status") == "success":
            fact_score = fact_result.get("similarity", 0)

            # Convert confidence % → decimal
            ml_score = confidence / 100

            final_score = (ml_score * 0.6) + (fact_score * 0.4)

            if final_score > 0.5:
                final_prediction = "REAL NEWS ✅ (Verified)"
            else:
                final_prediction = "FAKE NEWS ❌ (Low verification)"

        else:
            # fallback (your original logic remains)
            if trust == "High ✅" and "FAKE" in prediction:
                final_prediction = "LIKELY REAL (Trusted Source Override) ✅"
            else:
                final_prediction = prediction

        # LIME
        raw_explanation = get_explanation(text[:500])
        explanation = []
        for item in raw_explanation:
            word = str(item[0]).replace("np.str_('", "").replace("')", "")
            explanation.append(f"{word} : {item[1]}")

        # SHAP
        try:
            shap_explanation = get_shap_explanation(lr, tfidf, cleaned)
        except Exception as e:
            print("SHAP Error:", e)
            shap_explanation = []

        # SAVE
        save_prediction(session["user"], text, final_prediction, confidence)

    # FINAL CONCLUSION
    if final_prediction:
        if fact_result and fact_result.get("status") == "success":
            if "REAL" in final_prediction:
                conclusion = "The news is supported by both model prediction and real-world verification."
            else:
                conclusion = "The news appears suspicious despite some matches."

        elif fact_result and fact_result.get("status") == "no_data":
            conclusion = "This news could not be verified with trusted sources. It may be new or unverified."

        elif fact_result and fact_result.get("status") == "api_failed":
            conclusion = "Fact verification unavailable. Prediction is based only on the model."

        else:
            if "REAL" in final_prediction:
                conclusion = "The news appears reliable based on model prediction."
            else:
                conclusion = "The news may contain misleading or suspicious information."

    else:
        conclusion = None

    return render_template(
        "predict.html",
        prediction=prediction,
        final_prediction=final_prediction,
        explanation=explanation,
        shap_explanation=shap_explanation,
        confidence=confidence,
        conclusion=conclusion,
        user_input=user_input,
        trust=trust,
        fact_result=fact_result,
        entities=entities
    )


# Route to fetch live news using NewsAPI and display on the frontend
@app.route("/live-news")
def live_news():
    API_KEY = os.getenv("GNEWS_API_KEY")

    url = f"https://gnews.io/api/v4/top-headlines?country=in&lang=en&max=15&apikey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    articles = []

    # trusted keywords (flexible matching)
    trusted_keywords = ["bbc", "cnn", "reuters", "ndtv", "hindu", "times", "al jazeera", "cnbc"]

    if "articles" in data:
        for article in data["articles"]:
            title = article.get("title")
            link = article.get("url")
            source = article.get("source", {}).get("name", "Unknown")

            # Skip invalid entries
            if not title or not link:
                continue

            # Add only trusted sources
            if any(k in source.lower() for k in trusted_keywords):
                articles.append({
                    "title": title,
                    "url": link,
                    "source": source
                })

        # FALLBACK: if too few trusted news, add more (avoid empty page)
        if len(articles) < 5:
            for article in data["articles"]:
                title = article.get("title")
                link = article.get("url")
                source = article.get("source", {}).get("name", "Unknown")

                if not title or not link:
                    continue

                articles.append({
                    "title": title,
                    "url": link,
                    "source": source
                })

                if len(articles) >= 7:
                    break

    # LIMIT FINAL OUTPUT
    articles = articles[:7]

    return render_template("live_news.html", articles=articles)

#History route to display user's past predictions 
@app.route('/history')
def history():
    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    #conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT news, prediction, confidence
    FROM history
    WHERE username=?
    """, (session["user"],))

    user_history = cursor.fetchall()
    conn.close()

    return render_template("history.html", history=user_history)


#  UPDATED ADMIN ROUTE (MATPLOTLIB)
@app.route('/admin')
def admin():

    # FIXED session key
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))

    # FIXED DB PATH
    conn = sqlite3.connect(DB_PATH)
    #conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT username, news, prediction, confidence FROM history")
    history = cursor.fetchall()

    # Count REAL vs FAKE
    real_count = 0
    fake_count = 0

    for row in history:
        if "REAL" in row[2].upper():
            real_count += 1
        else:
            fake_count += 1

    conn.close()

    print("REAL:", real_count, "FAKE:", fake_count)

    #  GENERATE CHARTS USING MATPLOTLIB
    labels = ["Real News", "Fake News"]
    values = [real_count, fake_count]

    timestamp = int(time.time())

    bar_path = f"static/bar_{timestamp}.png"
    pie_path = f"static/pie_{timestamp}.png"

    # BAR CHART
    plt.figure()
    plt.bar(labels, values, color=['green', 'red'], edgecolor='black')
    plt.title("Real vs Fake News")
    plt.savefig(bar_path)
    plt.close()
    
    # PIE CHART
    plt.figure()
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
    plt.title("News Distribution")
    plt.savefig(pie_path)
    plt.close()

    plt.close('all')

    return render_template(
        "admin.html",
        history=history,
        bar_chart=bar_path,
        pie_chart=pie_path
    )

# USER PDF DOWNLOAD ROUTE
@app.route("/download_user_pdf")
def download_user_pdf():
    if "user" not in session:
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_PATH)
    #conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, news, prediction, confidence
        FROM history
        WHERE username=?
    """, (session["user"],))

    data = cursor.fetchall()
    conn.close()

    pdf = generate_pdf(data, "User Prediction History")

    return send_file(
        pdf,
        as_attachment=True,
        download_name="history.pdf",
        mimetype="application/pdf"
    )

# ADMIN PDF DOWNLOAD ROUTE
@app.route("/download_admin_pdf")
def download_admin_pdf():
    if "role" not in session or session["role"] != "admin":
        return redirect(url_for("login"))
    
    conn = sqlite3.connect(DB_PATH)
    #conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, news, prediction, confidence
        FROM history
    """)

    data = cursor.fetchall()
    conn.close()

    pdf = generate_pdf(data, "Admin - All Prediction History")

    return send_file(
       pdf,
       as_attachment=True,
       download_name="history.pdf",
       mimetype='application/pdf'
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# 1. Give the command to acivate the virtual environment : "venv\Scripts\activate"
# 2. Type in the teminal: "python app.py" to run the project
# 3. The admin login credentials are: username: admin, password: admin123


# SAMPLE NEWS ARTICLES/URLS FOR TESTING THE PROJECT :-

# Legendary playback singer Asha Bhosle passed away at the age of 92 on April 12, 2026, in Mumbai. 
# The cause of death was multi-organ failure following a cardiac arrest, after she was admitted to Breach Candy Hospital
# on April 11 with chest infection and extreme exhaustion. Her son, Anand Bhosle, and granddaughter, Zanai Bhosle, 
# confirmed the news.   --- REAL NEWS

# The Government of India has announced a new rule that all bank accounts without Aadhaar linkage will be 
# permanently frozen starting next month. Citizens are advised to immediately update their details to avoid 
# losing access to their funds. --- FAKE NEWS 

#https://www.ndtv.com/world-news/good-news-on-talks-with-iran-could-come-by-friday-trump-11394173?pfrom=home-ndtv_topscroll  ----Real News





