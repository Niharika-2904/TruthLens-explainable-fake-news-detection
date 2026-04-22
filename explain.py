from lime.lime_text import LimeTextExplainer
import numpy as np
import joblib

lr = joblib.load("models/logistic_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

class_names = ["FAKE", "REAL"]

def get_explanation(text):
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba(texts):
        cleaned = [t for t in texts]
        vectorized = tfidf.transform(cleaned)
        return lr.predict_proba(vectorized)

    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6
    )

    explanation_list = []

    for word, weight in exp.as_list():
        if weight > 0:
            impact = "This word increases the chance of REAL news"
        else:
            impact = "This word increases the chance of FAKE news"

        explanation_list.append((word, impact))

    return explanation_list