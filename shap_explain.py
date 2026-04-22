import shap
import numpy as np

def get_shap_explanation(model, vectorizer, text):

    X = vectorizer.transform([text])

    background = vectorizer.transform([
        "government policy news",
        "sports match result",
        "breaking news today"
    ])

    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X)

    feature_names = vectorizer.get_feature_names_out()
    values = shap_values[0]

    explanation = []

    for i in range(len(values)):
        if values[i] != 0:
            word = feature_names[i]
            score = round(values[i], 3)

            # Convert into readable sentence
            if score > 0:
                explanation.append(f"{word} : increases chance of REAL news")
            else:
                explanation.append(f"{word} : increases chance of FAKE news")

    return explanation[:8]