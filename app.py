import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(
    page_title="SkillGraph",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("skillgraph_model.joblib")

model = load_model()

st.title("📊 SkillGraph")
st.subheader("Job Market Intelligence & Skill-Gap Discovery")
st.write(
    "Explore job-market skill trends and predict the category "
    "of a job description using machine learning."
)

st.divider()

# Job category prediction
st.header("🔍 Job Category Predictor")

job_description = st.text_area(
    "Paste a job description",
    height=180,
    placeholder="Paste the full job description here..."
)

if st.button("Predict Category", type="primary"):
    if len(job_description.strip().split()) < 10:
        st.warning("Please enter a more detailed job description.")
    else:
        prediction = model.predict([job_description])[0]
        probabilities = model.predict_proba([job_description])[0]

        st.success(f"Predicted Category: {prediction}")

        probability_df = pd.DataFrame({
            "Category": model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Model probability scores")
        st.bar_chart(
            probability_df.set_index("Category")["Probability"]
        )

st.divider()

# Skill-demand analysis
st.header("📈 Skill Demand Analysis")
st.write(
    "Upload the original job-postings Parquet dataset to explore "
    "the most frequently listed skills."
)

uploaded_file = st.file_uploader(
    "Upload train-00000-of-00001.parquet",
    type=["parquet"]
)

if uploaded_file is not None:
    try:
        df = pd.read_parquet(uploaded_file)

        if "job_skill_set" not in df.columns:
            st.error("The uploaded file does not contain job_skill_set.")
        else:
            df["skills"] = df["job_skill_set"].apply(
                lambda value: value if isinstance(value, list)
                else __import__("ast").literal_eval(value)
            )

            all_skills = [
                skill.strip().lower()
                for skills in df["skills"]
                for skill in skills
                if isinstance(skill, str) and skill.strip()
            ]

            top_skills = Counter(all_skills).most_common(15)
            chart_df = pd.DataFrame(
                top_skills,
                columns=["Skill", "Mentions"]
            )

            st.subheader("Top 15 listed skills")
            st.bar_chart(chart_df.set_index("Skill"))

            st.caption(
                "Counts represent skill mentions in the uploaded dataset, "
                "not verified real-world hiring demand."
            )
    except Exception as error:
        st.error(f"Could not process the dataset: {error}")

st.divider()
st.caption("SkillGraph | DSML Project")
