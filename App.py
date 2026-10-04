from flask import Flask, render_template, request
import joblib
import os


# -----------------------------------
# BASE DIRECTORY
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------
# FLASK APP
# -----------------------------------

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------

model_path = os.path.join(
    BASE_DIR,
    "sentiment_model.pkl"
)

model = joblib.load(model_path)


# -----------------------------------
# HOME PAGE
# -----------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# -----------------------------------
# SENTIMENT PREDICTION
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get movie name
    movie = request.form.get(
        "movie",
        ""
    ).strip()

    # Get movie review
    review = request.form.get(
        "review",
        ""
    ).strip()


    # Check empty fields
    if not movie or not review:

        return render_template(
            "index.html",
            error="Please enter movie name and review."
        )


    # -----------------------------------
    # PREDICT SENTIMENT
    # -----------------------------------

    prediction = model.predict(
        [review]
    )[0]


    # -----------------------------------
    # CONFIDENCE
    # -----------------------------------

    probabilities = model.predict_proba(
        [review]
    )[0]

    confidence = max(
        probabilities
    ) * 100


    # -----------------------------------
    # RESULT
    # -----------------------------------

    if prediction == "positive":

        result = "Positive 😊"

    else:

        result = "Negative 😞"


    # -----------------------------------
    # SEND RESULT TO HTML
    # -----------------------------------

    return render_template(
        "index.html",

        movie=movie,

        review=review,

        result=result,

        confidence=round(
            confidence,
            2
        )
    )


# -----------------------------------
# RUN APPLICATION
# -----------------------------------

if __name__ == "__main__":

    print("--------------------------------")
    print("MOVIE REVIEW SENTIMENT ANALYSIS")
    print("--------------------------------")

    print(
        "App folder:",
        BASE_DIR
    )

    print(
        "Template folder:",
        os.path.join(
            BASE_DIR,
            "templates"
        )
    )

    print(
        "index.html exists:",
        os.path.exists(
            os.path.join(
                BASE_DIR,
                "templates",
                "index.html"
            )
        )
    )

    print("\nStarting Flask server...")

    app.run(
        debug=True
    )