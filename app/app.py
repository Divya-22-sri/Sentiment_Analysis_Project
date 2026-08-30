import streamlit as st
import joblib
import re
import os


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎬 IMDb Sentiment Analyzer")

st.write(
    "Analyze the sentiment of an IMDb movie review "
    "using Machine Learning."
)

st.divider()


# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------

st.subheader("📝 Enter Your Movie Review")

review = st.text_area(
    "Type your review below:",
    height=180,
    placeholder="Example: This movie was absolutely fantastic..."
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔮 Analyze Sentiment",
    use_container_width=True
):

    if not review.strip():

        st.warning(
            "Please enter a movie review first."
        )

    else:

        # Clean text
        cleaned_review = clean_text(review)

        # Convert text to TF-IDF
        review_vector = vectorizer.transform(
            [cleaned_review]
        )

        # Prediction
        prediction = model.predict(
            review_vector
        )[0]

        # Decision score
        decision_score = model.decision_function(
            review_vector
        )[0]

        # Convert score to approximate confidence
        confidence = (
            1 / (1 + abs(decision_score))
        )

        confidence_percentage = (
            (1 - confidence) * 100
        )


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")


        if prediction == 1:

            st.success(
                "😊 POSITIVE SENTIMENT"
            )

            st.write(
                f"Confidence Score: "
                f"{confidence_percentage:.2f}%"
            )

        else:

            st.error(
                "😞 NEGATIVE SENTIMENT"
            )

            st.write(
                f"Confidence Score: "
                f"{confidence_percentage:.2f}%"
            )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📌 About This Project")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Dataset Size",
        "49,582 Reviews"
    )

with col2:

    st.metric(
        "Model Accuracy",
        "91.02%"
    )


st.write(
    """
    **Machine Learning Model:** Tuned Linear SVM

    **Feature Extraction:** TF-IDF

    **Dataset:** IMDb Movie Reviews

    **Task:** Binary Sentiment Classification

    **Classes:** Positive and Negative
    """
)

st.caption(
    "Sentiment Analysis Project | Machine Learning + NLP"
)