# TruthLens – Explainable AI Fake News Detection Sysstem

## 📌 Overview

**TruthLens** is an AI-powered Fake News Detection web application built using **Flask**, **Machine Learning**, and **Natural Language Processing (NLP)** techniques.
The system analyzes news articles or headlines and predicts whether the news is **Real** or **Fake** with confidence scores and explainable AI insights.

The project also includes:

* User Authentication
* Prediction History
* Admin Dashboard
* Explainable AI (LIME & SHAP)
* PDF Report Generation
* Live News Verification
* URL-based News Extraction



## 🚀 Features

### 🔐 Authentication & Authorization

* User Registration & Login
* Password Hashing for Security
* Role-Based Access:

  * User
  * Admin

### 📰 Fake News Detection

* Detects whether news is **Real** or **Fake**
* Uses:

  * TF-IDF Vectorization
  * Logistic Regression Model

### 📊 Confidence Score

* Displays prediction confidence percentage
* Generates Trust Score

### 🧠 Explainable AI

* LIME Explanation
* SHAP Explanation
* Human-readable reasoning for predictions

### 🌐 URL-Based News Extraction

* Extracts article content directly from news URLs

### 🗂 Prediction History

* Saves all user predictions in SQLite Database
* Users can view their previous analyses

### 👨‍💼 Admin Dashboard

* View all users’ prediction history
* Graphical analytics using Matplotlib

### 📄 PDF Report Generation

* Download prediction reports as PDF

### 📰 Live News Verification

* Fetches live news articles using News APIs
* Verifies authenticity of trending news



## 🛠️ Tech Stack

### Frontend

* HTML
* CSS
* Bootstrap
* JavaScript

### Backend

* Python
* Flask

### Machine Learning & NLP

* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* LIME
* SHAP

### Database

* SQLite

### Visualization

* Matplotlib

### PDF Generation

* ReportLab



## 📂 Project Structure

```bash
TruthLens/
│
├── app.py
├── model/
│   ├── fake_news_model.pkl
│   ├── vectorizer.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── predict.html
│   ├── history.html
│   ├── admin.html
│
├── database/
│   ├── truthlens.db
│
├── utils/
│   ├── preprocess.py
│   ├── explain.py
│   ├── extract.py
│
├── reports/
│
├── requirements.txt
└── README.md
```



## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/TruthLens.git
cd TruthLens
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python app.py
```

### 5️⃣ Open in Browser

```bash
http://127.0.0.1:5000/
```



## 🤖 Machine Learning Workflow

1. Dataset Collection
2. Data Cleaning & Preprocessing
3. TF-IDF Feature Extraction
4. Model Training using Logistic Regression
5. Prediction Generation
6. Explainability using LIME & SHAP



## 📈 Model Details

| Component            | Technique Used      |
| -------------------- | ------------------- |
| Text Vectorization   | TF-IDF              |
| Classification Model | Logistic Regression |
| Explainability       | LIME & SHAP         |
| Database             | SQLite              |



## 🔍 How It Works

1. User enters:

   * News headline
   * News article
   * OR news URL

2. System preprocesses the text

3. TF-IDF converts text into numerical vectors

4. Logistic Regression predicts:

   * Real News
   * Fake News

5. System displays:

   * Prediction Result
   * Confidence Score
   * Trust Score
   * AI Explanation



## 🔐 Security Features

* Password Hashing
* Session Management
* Role-Based Access Control
* Input Validation


  
## 📄 Requirements

Example dependencies:

```txt
Flask
scikit-learn
numpy
pandas
matplotlib
lime
shap
reportlab
newspaper3k
beautifulsoup4
requests
```



## 👩‍💻 Author

**Niharika Saxena**

B.Tech CSE Student

