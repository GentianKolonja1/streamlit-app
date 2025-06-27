import os 
import json # for reading json files
import streamlit as st # for creating the web app
import joblib # for loading the model

model_path = "model.joblib"
vect_path  = "vectorizer.joblib"

if not os.path.exists(model_path) or not os.path.exists(vect_path):
    st.error("Model or vectorizer not found. Please run `python ai_grader.py` first.")
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vect_path)

metrics = {}
if os.path.exists("metrics.json"):
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
# Streamlit UI 
st.set_page_config(page_title="AI Essay Grader", layout="centered")
st.title("C Essay Grader")
st.write("Paste an essay or upload a `.txt` to get a predicted grade.")


essay_input = st.text_area("Paste your essay here:", height=300)
uploaded     = st.file_uploader("Or upload a .txt file", type="txt")
if uploaded and not essay_input:
    essay_input = uploaded.read().decode("utf-8")
    st.success(" Essay loaded from file")

st.write("---")

# Grade button
if st.button("Grade Essay"):
    if not essay_input.strip():
        st.warning("Please enter or upload an essay.")
    else:
        # Vectorize & predict
        X = vectorizer.transform([essay_input])
        raw_score = model.predict(X)[0]
        rounded   = int(round(raw_score))
        rounded   = max(min(rounded, 6), 0)  # clamp to [0,6]

        # Display scores
        st.metric("Raw Score",      f"{raw_score:.2f} / 6")
        st.metric("Rounded Score",  f"{rounded} / 6")

        # Simple feedback
        if rounded <= 2:
            st.warning(" Needs stronger arguments and structure.")
        elif rounded <= 4:
            st.info("Consider deeper analysis and evidence.")
        else:
            st.success(" Excellent! Clear arguments and well done :) ")

        # Show model metrics if available
        if metrics:
            st.write("---")
            st.subheader("Model Performance on Test Data")
            st.write(f"- R² Score: **{metrics.get('r2', 0):.2f}**")
            st.write(f"- RMSE: **{metrics.get('rmse', 0):.2f}**")
            st.write(f"- Exact-match Accuracy: **{metrics.get('accuracy', 0):.1f}%**")

st.write("\n---\n*Built with TF-IDF + Random Forest. ")
