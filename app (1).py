import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SkillGraph | Job Market Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN SYSTEM ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ivory: #F8F7F3;
    --navy: #202944;
    --violet: #8171B5;
    --muted: #777C8D;
    --line: #E8E5DF;
}

.stApp {
    background: var(--ivory);
    color: var(--navy);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--navy) !important;
    letter-spacing: -0.7px;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1440px;
}

section[data-testid="stSidebar"] {
    background: #F0EEE8;
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--navy) !important;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 19px 20px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
}

div[data-testid="stMetricValue"] {
    color: var(--navy);
    font-family: 'Manrope', sans-serif;
}

div.stButton > button {
    background: var(--navy);
    color: white;
    border: 1px solid var(--navy);
    border-radius: 9px;
    padding: 0.55rem 1.15rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background: var(--violet);
    color: white;
    border-color: var(--violet);
}

div[data-testid="stTextArea"] textarea {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 10px;
}

hr {
    border-color: var(--line);
}

.sg-eyebrow {
    color: var(--violet);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.sg-hero {
    background: linear-gradient(120deg, #FFFFFF 0%, #F0EDF8 100%);
    border: 1px solid #E8E2F2;
    border-radius: 20px;
    padding: 35px;
    margin: 12px 0 25px 0;
}

.sg-hero h1 {
    font-size: 2.45rem;
    line-height: 1.2;
    margin: 12px 0;
}

.sg-hero p {
    color: #686D7F;
    font-size: 1rem;
    max-width: 650px;
    line-height: 1.8;
}

.sg-note {
    color: var(--muted);
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
MODEL_PATH = Path(__file__).with_name("skillgraph_model.joblib")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.8rem;font-weight:800;color:#202944;'>"
        "◈ SkillGraph</div>",
        unsafe_allow_html=True
    )
    st.caption("JOB MARKET INTELLIGENCE")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "Job Predictor", "Skill Explorer", "Skill Gap"],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.caption("Machine learning · Skill analytics")
    st.markdown(
        "<div class='sg-note'>Designed for data-driven career discovery.</div>",
        unsafe_allow_html=True
    )

# ---------- OVERVIEW ----------
if page == "Overview":
    st.markdown(
        "<div class='sg-eyebrow'>YOUR CAREER INTELLIGENCE WORKSPACE</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="sg-hero">
        <div class="sg-eyebrow">WELCOME TO SKILLGRAPH</div>
        <h1>Understand the market.<br>Build the right skills.</h1>
        <p>
            Explore skills mentioned in job postings, discover career
            patterns and use machine learning to understand job categories.
            Turn job-market data into more informed career decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Your workspace")
    st.write("Choose an area to start exploring.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 01 · Job intelligence")
        st.write("Predict a job's category from its description.")
        if st.button("Open job predictor", use_container_width=True):
            st.session_state["page_choice"] = "Job Predictor"
            st.rerun()

    with c2:
        st.markdown("#### 02 · Skill explorer")
        st.write("Explore frequently listed skills and market patterns.")
        if st.button("Explore skills", use_container_width=True):
            st.session_state["page_choice"] = "Skill Explorer"
            st.rerun()

    with c3:
        st.markdown("#### 03 · Skill gap")
        st.write("Compare your current skills with a target career.")
        if st.button("Analyze skill gap", use_container_width=True):
            st.session_state["page_choice"] = "Skill Gap"
            st.rerun()

    st.divider()
    st.subheader("Built for better career decisions")
    a, b = st.columns(2)

    with a:
        st.markdown("**Evidence over guesswork**")
        st.write(
            "Skill insights are based on the job-postings dataset "
            "rather than invented market statistics."
        )

    with b:
        st.markdown("**Machine learning, made useful**")
        st.write(
            "A trained text-classification model connects job "
            "descriptions to one of five job categories."
        )

    if model is None:
        st.warning(
            "The model file is not beside app.py yet. Add "
            "skillgraph_model.joblib to enable predictions."
        )

# ---------- JOB PREDICTOR ----------
elif page == "Job Predictor":
    st.markdown("<div class='sg-eyebrow'>MACHINE LEARNING</div>",
                unsafe_allow_html=True)
    st.title("Job category predictor")
    st.write(
        "Paste a job description to estimate which category it belongs to."
    )

    if model is None:
        st.error(
            "Model not found. Place skillgraph_model.joblib "
            "in the same folder as app.py."
        )
    else:
        description = st.text_area(
            "Job description",
            height=220,
            placeholder=(
                "Paste the responsibilities, qualifications and "
                "requirements from a job posting..."
            )
        )

        if st.button("Analyze job description", type="primary"):
            if len(description.strip().split()) < 10:
                st.warning("Please enter at least 10 words for a more useful test.")
            else:
                prediction = model.predict([description])[0]
                probabilities = model.predict_proba([description])[0]

                st.success(f"Predicted category: {prediction}")

                probability_df = pd.DataFrame({
                    "Category": model.classes_,
                    "Model score": probabilities
                }).sort_values("Model score", ascending=False)

                st.subheader("Model scores")
                st.bar_chart(
                    probability_df.set_index("Category"),
                    horizontal=True
                )
                st.caption(
                    "Scores reflect the model's relative estimates, "
                    "not guaranteed correctness."
                )

# ---------- SKILL EXPLORER ----------
elif page == "Skill Explorer":
    st.markdown("<div class='sg-eyebrow'>JOB MARKET ANALYTICS</div>",
                unsafe_allow_html=True)
    st.title("Skill explorer")
    st.write("Explore skills listed in your job-postings dataset.")

    uploaded = st.file_uploader(
        "Upload the dataset in Parquet format",
        type=["parquet"]
    )

    if uploaded is not None:
        try:
            import ast
            from collections import Counter

            data = pd.read_parquet(uploaded)

            if not {"job_skill_set", "category"}.issubset(data.columns):
                st.error("The dataset needs job_skill_set and category columns.")
            else:
                def parse_skills(value):
                    if isinstance(value, list):
                        return value
                    if isinstance(value, str):
                        parsed = ast.literal_eval(value)
                        return parsed if isinstance(parsed, list) else []
                    return []

                data["skills"] = data["job_skill_set"].apply(parse_skills)
                total_mentions = sum(
                    len([s for s in skills if isinstance(s, str) and s.strip()])
                    for skills in data["skills"]
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Job postings", f"{len(data):,}")
                m2.metric("Job categories", data["category"].nunique())
                m3.metric("Skill mentions", f"{total_mentions:,}")

                counts = Counter(
                    skill.strip().lower()
                    for skills in data["skills"]
                    for skill in skills
                    if isinstance(skill, str) and skill.strip()
                )

                top_n = st.slider("Skills to display", 5, 20, 15)
                top_skills = pd.DataFrame(
                    counts.most_common(top_n),
                    columns=["Skill", "Job-posting mentions"]
                )

                st.subheader("Most frequently listed skills")
                st.bar_chart(
                    top_skills.set_index("Skill"),
                    horizontal=True
                )

                categories = sorted(data["category"].dropna().unique())
                selected = st.selectbox("Compare a job category", categories)

                category_counts = Counter(
                    skill.strip().lower()
                    for skills in data.loc[
                        data["category"] == selected, "skills"
                    ]
                    for skill in skills
                    if isinstance(skill, str) and skill.strip()
                )

                category_df = pd.DataFrame(
                    category_counts.most_common(top_n),
                    columns=["Skill", "Mentions"]
                )
                st.subheader(f"Frequently listed skills · {selected}")
                st.bar_chart(category_df.set_index("Skill"), horizontal=True)

                st.caption(
                    "Counts measure mentions in this dataset. They do not "
                    "represent the entire job market or guarantee demand."
                )

        except Exception as error:
            st.error(f"Unable to read the dataset: {error}")
    else:
        st.info(
            "Upload the same Parquet dataset you analyzed in Colab "
            "to populate these charts."
        )

# ---------- SKILL GAP ----------
elif page == "Skill Gap":
    st.markdown("<div class='sg-eyebrow'>CAREER DEVELOPMENT</div>",
                unsafe_allow_html=True)
    st.title("Skill-gap analyzer")
    st.write(
        "Compare skills you already have with skills listed for a target role."
    )

    st.info(
        "This first version uses the skills in your uploaded dataset "
        "as a reference. It is not a hiring prediction."
    )

    uploaded = st.file_uploader(
        "Upload the job-postings Parquet dataset",
        type=["parquet"],
        key="gap_dataset"
    )

    current_skills = st.text_input(
        "Your current skills, separated by commas",
        placeholder="Python, SQL, Excel, communication"
    )

    if uploaded is not None:
        try:
            import ast
            from collections import Counter

            data = pd.read_parquet(uploaded)

            if not {"job_skill_set", "category"}.issubset(data.columns):
                st.error("The dataset needs job_skill_set and category columns.")
            else:
                def parse_gap_skills(value):
                    if isinstance(value, list):
                        return value
                    if isinstance(value, str):
                        parsed = ast.literal_eval(value)
                        return parsed if isinstance(parsed, list) else []
                    return []

                data["skills"] = data["job_skill_set"].apply(parse_gap_skills)
                role = st.selectbox(
                    "Target job category",
                    sorted(data["category"].dropna().unique())
                )

                if st.button("Find potential skill gaps", type="primary"):
                    role_skills = Counter(
                        skill.strip().lower()
                        for skills in data.loc[
                            data["category"] == role, "skills"
                        ]
                        for skill in skills
                        if isinstance(skill, str) and skill.strip()
                    )

                    owned = {
                        item.strip().lower()
                        for item in current_skills.split(",")
                        if item.strip()
                    }

                    if not role_skills:
                        st.warning("No skills were found for this category.")
                    else:
                        reference = role_skills.most_common(15)
                        matched = [
                            (skill, count) for skill, count in reference
                            if skill in owned
                        ]
                        missing = [
                            (skill, count) for skill, count in reference
                            if skill not in owned
                        ]

                        left, right = st.columns(2)
                        with left:
                            st.subheader("Skills you listed")
                            if matched:
                                for skill, _ in matched:
                                    st.write(f"✓ {skill}")
                            else:
                                st.write("No direct matches among the top 15.")

                        with right:
                            st.subheader("Skills to explore")
                            if missing:
                                for skill, _ in missing:
                                    st.write(f"＋ {skill}")
                            else:
                                st.write("You listed all skills in this reference set.")

                        st.caption(
                            "Reference: the 15 most frequently mentioned skills "
                            "for this category in the uploaded dataset. "
                            "A missing skill is not proof that you lack it."
                        )
        except Exception as error:
            st.error(f"Unable to analyze the dataset: {error}")
    else:
        st.info("Upload the dataset to compare your skills with a target role.")

# ---------- FOOTER ----------
st.divider()
st.caption("SKILLGRAPH  ·  Job Market Intelligence & Skill-Gap Discovery")
