"""
StressLens AI — AI Exam Anxiety Detector
Complete Streamlit App — trains model AND shows UI in one file
No Flask, No CORS, No integration issues.
"""

import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import joblib
import nltk

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Constants ─────────────────────────────────────────────────
PHYSIO_FEATURES = [
    "heart_rate", "sleep_hours", "study_hours_per_day",
    "days_since_last_break", "social_interactions_per_week",
    "water_intake_liters", "exercise_minutes_per_day",
]
LABELS    = ["Low", "Moderate", "High"]
MODEL_DIR = "models"
DATA_PATH = os.path.join("dataset", "exam_anxiety.csv")

STOPWORDS = set(stopwords.words("english"))
stemmer   = PorterStemmer()

# ── Text cleaning ──────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [stemmer.stem(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# ── Dataset generator ──────────────────────────────────────────
def generate_dataset():
    import random
    random.seed(42); np.random.seed(42)

    LOW = [
        "I feel relaxed about my upcoming exams",
        "I am confident and calm about exams",
        "I sleep well and feel mentally prepared",
        "I manage my time well and feel in control",
        "I feel okay about the exams nothing unusual",
        "I enjoy studying and feel ready for assessments",
        "No major worries just some normal exam jitters",
        "I feel good about my chances in the exam",
    ]
    MOD = [
        "I am somewhat nervous and find it hard to concentrate",
        "I worry about failing but I keep studying",
        "I feel stressed but manage it with breaks",
        "Sometimes I cannot sleep thinking about exams",
        "I feel overwhelmed when I look at the syllabus",
        "I get anxious during revision but calm down after",
        "I feel moderate pressure from family expectations",
        "I sometimes procrastinate due to exam pressure",
    ]
    HIG = [
        "I cannot stop worrying about failing the exam",
        "I feel extremely stressed and cannot focus at all",
        "I experience panic attacks thinking about exams",
        "I have not slept for days because of exam fear",
        "I feel completely hopeless and unprepared for exams",
        "My heart races and I feel nauseous about exams",
        "I cry frequently because of unbearable exam pressure",
        "My mind goes blank whenever I try to study",
    ]

    rows = []
    for label, code, phrases, hr_m, sl_m, st_m, br, si, wi, ex in [
        ("Low",      0, LOW, 72, 7.5, 5, (0,2),  (5,10), 2.5, (20,60)),
        ("Moderate", 1, MOD, 83, 6.0, 7, (2,5),  (2,6),  1.8, (5,25)),
        ("High",     2, HIG, 96, 4.5, 9, (5,14), (0,2),  1.2, (0,10)),
    ]:
        for _ in range(400):
            rows.append({
                "text_response":               random.choice(phrases),
                "anxiety_label":               label,
                "anxiety_code":                code,
                "heart_rate":                  round(np.random.normal(hr_m, 6), 1),
                "sleep_hours":                 round(np.random.normal(sl_m, 0.7), 1),
                "study_hours_per_day":         round(np.random.normal(st_m, 1), 1),
                "days_since_last_break":       random.randint(br[0], br[1]),
                "social_interactions_per_week":random.randint(si[0], si[1]),
                "water_intake_liters":         round(np.random.normal(wi, 0.3), 1),
                "exercise_minutes_per_day":    random.randint(ex[0], ex[1]),
            })

    os.makedirs("dataset", exist_ok=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    return df

# ── Train model ────────────────────────────────────────────────
def train_and_save_model():
    # Load or generate dataset
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dataset()

    X_text = df["text_response"].apply(clean_text)
    X_num  = df[PHYSIO_FEATURES].fillna(df[PHYSIO_FEATURES].median())
    y      = df["anxiety_code"]

    # TF-IDF
    tfidf       = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    text_features = tfidf.fit_transform(X_text)

    # Scaler
    scaler       = StandardScaler()
    num_features = scaler.fit_transform(X_num)
    num_sparse   = sp.csr_matrix(num_features)

    # Combined features
    X_combined = sp.hstack([text_features, num_sparse])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train all 3 models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        results[name] = {"model": m, "accuracy": acc}

    # Naive Bayes (needs non-negative)
    X_train_nb = sp.csr_matrix(X_train.toarray() - X_train.toarray().min())
    X_test_nb  = sp.csr_matrix(X_test.toarray()  - X_train.toarray().min())
    nb = MultinomialNB()
    nb.fit(X_train_nb, y_train)
    results["Naive Bayes"] = {
        "model": nb,
        "accuracy": accuracy_score(y_test, nb.predict(X_test_nb))
    }

    # Best model = Random Forest
    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(tfidf,      f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")

    return results, best_name

# ── Load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        tfidf  = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        return model, tfidf, scaler, True
    except:
        return None, None, None, False

# ── Predict ────────────────────────────────────────────────────
def predict(text, physio_vals, model, tfidf, scaler):
    cleaned  = clean_text(text)
    text_vec = tfidf.transform([cleaned])
    num_vec  = scaler.transform([physio_vals])
    num_sp   = sp.csr_matrix(num_vec)
    X        = sp.hstack([text_vec, num_sp])
    code     = int(model.predict(X)[0])
    proba    = model.predict_proba(X)[0]
    return LABELS[code], proba

# ── Recommendations ────────────────────────────────────────────
RECS = {
    "Low": {
        "color": "#cc922e", "emoji": "😊", "bg": "#d8d4ed",
        "message": "You are managing exam pressure well! 🌟",
        "tips": [
            "✅ Maintain your current study schedule",
            "✅ Keep up regular sleep and exercise habits",
            "✅ Practice mindfulness to stay grounded",
            "✅ Reward yourself for consistent preparation",
        ]
    },
    "Moderate": {
        "color": "#f39c12", "emoji": "😐", "bg": "#fff3cd",
        "message": "You are experiencing manageable exam stress. ⚡",
        "tips": [
            "⏱️ Try the Pomodoro technique (25 min study, 5 min break)",
            "🧘 Practice deep breathing exercises daily",
            "☕ Reduce caffeine and improve sleep hygiene",
            "👥 Talk to a friend or family about your concerns",
            "📋 Break syllabus into smaller daily targets",
        ]
    },
    "High": {
        "color": "#e74c3c", "emoji": "😰", "bg": "#f8d7da",
        "message": "You may be under significant exam anxiety. Please seek support. 🆘",
        "tips": [
            "🏥 Speak to your college counselor immediately",
            "🧘 Practice progressive muscle relaxation",
            "📵 Take a complete break from studies for one day",
            "📞 Call iCall helpline: 9152987821",
            "💙 Remember: one exam does not define your future",
        ]
    }
}

# ══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="StressLens AI — AI Anxiety Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #f0f4ff; }
  .block-container { padding-top: 2rem; }
  .header-box {
    background: linear-gradient(135deg, #3a0ca3, #4361ee, #7209b7);
    padding: 2rem; border-radius: 16px; text-align: center;
    color: white; margin-bottom: 2rem;
  }
  .header-box h1 { font-size: 2.4rem; font-weight: 800; margin: 0; }
  .header-box p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }
  .result-box {
    padding: 1.5rem 2rem; border-radius: 14px;
    text-align: center; margin: 1rem 0;
  }
  .tip-box {
    background: #f8faff; border-radius: 10px;
    padding: 1rem 1.5rem; border-left: 5px solid #4361ee;
    margin-top: 1rem;
  }
  .stButton > button {
    background: linear-gradient(135deg, #4361ee, #7209b7) !important;
    color: white !important; font-size: 1.1rem !important;
    font-weight: 700 !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; border: none !important;
    width: 100% !important;
  }
  div[data-testid="metric-container"] {
    background: white; border-radius: 10px;
    padding: 0.8rem; border: 1px solid #e0e7ff;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🧠 StressLens AI</h1>
  <p>AI-Powered Exam Anxiety Detection System | B.Tech Project</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Train Model ──────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Model Control Panel")
    st.markdown("---")

    model, tfidf, scaler, loaded = load_model()

    if loaded:
        st.success("✅ Model is loaded & ready!")
    else:
        st.error("❌ Model not trained yet")

    st.markdown("---")
    st.subheader("🎓 Train Model")
    st.write("Click below to generate dataset and train all 3 ML models.")

    if st.button("🚀 Train Model Now"):
        with st.spinner("Training in progress... please wait ⏳"):
            try:
                results, best = train_and_save_model()
                st.success(f"✅ Training complete!\n🏆 Best: {best}")
                st.markdown("### 📊 Model Accuracies")
                for name, res in results.items():
                    st.metric(name, f"{res['accuracy']:.2%}")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")
    st.markdown("**Stack:** Python · Scikit-learn · Streamlit")
    st.markdown("**Models:** LR · RF · NB")
    st.markdown("**Input:** Text + Physiological")
    st.caption("⚠️ For educational purposes only.")

# ── Main Form ──────────────────────────────────────────────────
if not loaded:
    st.warning("⚠️ Model not trained yet. Click **Train Model Now** in the left sidebar to get started.")
    st.stop()

st.subheader("📝 Self-Assessment Form")
st.write("Answer honestly — your responses are never stored.")

# Text input
text_input = st.text_area(
    "💬 How do you feel about your upcoming exams?",
    placeholder="E.g., I feel extremely stressed and cannot sleep properly...",
    height=120
)

st.markdown("---")
st.subheader("📊 Physiological & Lifestyle Metrics")
st.caption("Fill in your current stats as accurately as possible.")

# Physiological inputs — 2 columns
col1, col2 = st.columns(2)

with col1:
    heart_rate     = st.number_input("❤️ Resting Heart Rate (bpm)",    min_value=40,  max_value=150, value=75)
    sleep_hours    = st.number_input("😴 Sleep Hours per Night",        min_value=0.0, max_value=12.0, value=6.5, step=0.5)
    study_hours    = st.number_input("📚 Study Hours per Day",          min_value=0.0, max_value=20.0, value=7.0, step=0.5)
    days_no_break  = st.number_input("🗓️ Days Since Last Full Break",   min_value=0,   max_value=60,  value=3)

with col2:
    social         = st.number_input("🤝 Social Interactions / Week",   min_value=0,   max_value=30,  value=4)
    water          = st.number_input("💧 Water Intake (liters/day)",    min_value=0.0, max_value=6.0, value=2.0, step=0.1)
    exercise       = st.number_input("🏃 Exercise (minutes/day)",       min_value=0,   max_value=180, value=15)

st.markdown("---")

# ── Predict Button ─────────────────────────────────────────────
if st.button("🔍 Analyze My Anxiety Level"):
    if not text_input.strip():
        st.error("⚠️ Please describe how you feel about your exams in the text box above.")
    else:
        with st.spinner("🤖 AI is analyzing your responses..."):
            physio_vals = [
                heart_rate, sleep_hours, study_hours,
                days_no_break, social, water, exercise
            ]

            try:
                label, proba = predict(
                    text_input, physio_vals, model, tfidf, scaler
                )
                rec = RECS[label]

                # ── Result banner ──────────────────────────────
                st.markdown("---")
                st.subheader("📊 Your Assessment Results")

                st.markdown(f"""
                <div class="result-box" style="background:{rec['bg']};
                     border: 2px solid {rec['color']};">
                  <div style="font-size:3.5rem">{rec['emoji']}</div>
                  <div style="font-size:1.8rem; font-weight:800;
                       color:{rec['color']}">{label} Anxiety</div>
                  <div style="font-size:1rem; color:#555; margin-top:6px">
                    AI Confidence: <b>{proba[LABELS.index(label)]*100:.1f}%</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability bars ───────────────────────────
                st.markdown("#### 📈 Probability Distribution")
                colors = {"Low": "#2e75cc", "Moderate": "#f39c12", "High": "#e74c3c"}
                for i, lbl in enumerate(LABELS):
                    pct = proba[i] * 100
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"""
                        <div style="margin-bottom:8px">
                          <div style="font-size:0.85rem;font-weight:600;
                               margin-bottom:3px">{lbl}</div>
                          <div style="background:#e9ecef;border-radius:100px;
                               height:14px;overflow:hidden">
                            <div style="width:{pct}%;background:{colors[lbl]};
                                 height:100%;border-radius:100px;
                                 transition:width 1s"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div style='padding-top:16px;font-weight:700;"
                                    f"color:{colors[lbl]}'>{pct:.1f}%</div>",
                                    unsafe_allow_html=True)

                # ── Recommendations ────────────────────────────
                st.markdown("#### 💡 Personalized Recommendations")
                st.markdown(f"""
                <div class="tip-box" style="border-left-color:{rec['color']}">
                  <b style="font-size:1rem;color:{rec['color']}">{rec['message']}</b>
                  <ul style="margin-top:10px;padding-left:20px">
                    {''.join(f'<li style="margin-bottom:6px">{t}</li>' for t in rec['tips'])}
                  </ul>
                </div>
                """, unsafe_allow_html=True)

                # ── Summary metrics ────────────────────────────
                st.markdown("#### 📋 Your Input Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("❤️ Heart Rate",   f"{heart_rate} bpm")
                m2.metric("😴 Sleep",         f"{sleep_hours} hrs")
                m3.metric("📚 Study",         f"{study_hours} hrs/day")
                m4.metric("🏃 Exercise",      f"{exercise} min/day")

            except Exception as e:
                st.error(f"Prediction error: {e}")