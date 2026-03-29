import streamlit as st
import fitz
import docx
import sqlite3
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
import pandas as pd
import os
st.set_page_config(page_title="Auto Assignment Evaluation", layout="wide")
# change background color for page
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)
st.markdown(
    "<h1 style='color:#008208;font-weight:900'>📘 Auto Assignment Evaluation System</h1>",
    unsafe_allow_html=True
    )
# st.title("📘 Auto Assignment Evaluation System")
st.caption("Assignment Evaluation (PDF / DOCX / TXT)")
st.divider()
# ---------------- DATABASE ----------------
conn = sqlite3.connect("evaluation.db", check_same_thread=False)
c = conn.cursor()

# Create table (initial)
c.execute("""
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    model_answer TEXT
)
""")

# ADD COLUMN IF NOT EXISTS (MIGRATION)
try:
    c.execute("ALTER TABLE assignments ADD COLUMN instructions TEXT")
except:
    pass  # already exists

conn.commit()

# ---------------- SBERT MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sbert = load_model()

# ---------------- FUNCTION to Extract Answer from User Side----------------
def extract_text(file):
    file_bytes = file.read()

    if file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        return " ".join([p.get_text() for p in pdf])

    elif file.name.endswith(".docx"):
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    elif file.name.endswith(".txt"):
        return file_bytes.decode("utf-8")

    return ""

def similarity(a, b):
    return util.cos_sim(
        sbert.encode(a, convert_to_tensor=True),
        sbert.encode(b, convert_to_tensor=True)
    ).item()

def keyword_score(text, keywords):
    if not keywords:
        return 0
    count = sum(1 for k in keywords if k.lower() in text.lower())
    return count / len(keywords)
# allot marks
def marks(score):
    if score >= 0.95: 
        return 10
    elif score >= 0.90: return 9.5
    elif score >= 0.85: return 9
    elif score >= 0.80: return 8.5
    elif score >= 0.75: return 8
    elif score >= 0.70: return 7
    elif score >= 0.65: return 6.5
    elif score >= 0.60: return 6
    elif score >= 0.55: return 5.5
    elif score >= 0.50: return 5
    else: return 0

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "result" not in st.session_state:
    st.session_state.result = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

def go(page):
    st.session_state.page = page
    st.rerun()
# custom CSS code for changing background color and foreground color for buttons
st.markdown("""
<style>

/* LOGIN BUTTON - GREEN */
div.stButton > button:first-child {
    background-color: #6a3be4;
    color: white;
    font-weight: bold;
}

/* SUBMIT BUTTON - RED */
div.stButton:nth-of-type(3) > button {
    background-color: #dc3545;
    color: white;
    font-weight: bold;
}

/* Hover effects */
div.stButton > button:hover {
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)
# ---------------- HOME ----------------
def home():
    if st.button("Student"):
        go("student")
    if st.button("Faculty"):
        go("faculty")
    if st.button("Download Samples"):
        go("samples")

# ---------------- FACULTY ----------------
def faculty():
    st.title("👨‍🏫 Faculty Panel")

    title = st.text_input("Assignment Title")

    instructions = st.text_area(
        "General Instructions",
        """1. Requested to submit assignment with on time
2. Acceptable formats: .pdf, .docx, .txt
3. Maximum mark: 10"""
    )

    file = st.file_uploader("Upload Model Answer", type=["pdf","docx","txt"])

    if st.button("Post Assignment"):
        if file:
            with st.spinner("Posting Assignment..."):
                text = extract_text(file)

                c.execute("""
                    INSERT INTO assignments (title, model_answer, instructions)
                    VALUES (?, ?, ?)
                """, (title, text, instructions))

                conn.commit()

            st.success("Assignment is Posted Successfully.")

    if st.button("Back"):
        go("home")

# ---------------- STUDENT ----------------
def student():
    st.title("🎓 Student Panel")

    c.execute("SELECT * FROM assignments ORDER BY id DESC LIMIT 1")
    data = c.fetchone()

    name = st.text_input("Name")

    if data:
        st.text_input("Assignment", value=data[1], disabled=True)

        # handle old rows without instructions
        instr = data[3] if len(data) > 3 and data[3] else "No instructions provided"
        st.text_area("Instructions", value=instr, disabled=True)

        model_text = data[2]
    else:
        st.warning("No assignment available")
        return

    keywords = st.text_input("Keywords (comma separated)")
    file = st.file_uploader("Upload Answer", type=["pdf","docx","txt"])

    if file is not None:
        if st.session_state.last_file != file.name:
            st.session_state.result = None
            st.session_state.last_file = file.name

    # RED BUTTON
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: red;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Evaluate"):
        if file:
            with st.spinner("Evaluating..."):
                student_text = extract_text(file)

                sim = similarity(student_text, model_text)
                key_list = [k.strip() for k in keywords.split(",")] if keywords else []
                key_val = keyword_score(student_text, key_list)

                if sim < 0.5:
                    final = 0
                    mark = 0
                    status = "❌ Mismatch (Different Topic)"
                else:
                    final = sim*0.95 + key_val*0.05
                    mark = marks(final)
                    status = "✅ Evaluated"

                st.session_state.result = {
                    "student": student_text,
                    "model": model_text,
                    "sim": sim,
                    "key": key_val,
                    "final": final,
                    "marks": mark,
                    "status": status
                }

    # RESULT
    if st.session_state.result:
        r = st.session_state.result

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
    "<h4 style='color:#338d11;font-weight:900'>Student Answer</h2>",
    unsafe_allow_html=True
    )
            st.text_area("", r["student"], height=500, key=f"s_{r['marks']}")

        with col2:
            st.markdown(
    "<h4 style='color:#338d11;font-weight:900'>Faculty Answer</h2>",
    unsafe_allow_html=True
    )
            st.text_area("", r["model"], height=500, key=f"m_{r['marks']}")

        st.success(r["status"])

        st.write(f"Semantic Score : {r['sim']*100:.2f}%")
        st.write(f"Keyword Score : {r['key']*100:.2f}%")
        st.write(f"Final Score : {r['final']*100:.2f}%")
        st.write(f"🏆 Marks : {r['marks']}/10")

        # CHART
        df = pd.DataFrame({
            "Criteria": ["Semantic", "Keywords"],
            "Score": [
                r["sim"]*100,
                r["key"]*100
            ]
        })

        st.subheader("📊 Score Breakdown")
        st.bar_chart(df.set_index("Criteria"))

    if st.button("Back"):
        go("home")
# ---------------- SAMPLES ----------------
def samples():
    st.title("📥 Sample Assignments")
    folder = "sample"
    if os.path.exists(folder):
        for f in os.listdir(folder):
            with open(os.path.join(folder, f), "rb") as file:
                st.download_button(f"Download {f}", file, file_name=f)
    if st.button("Back"):
        go("home")
# ---------------- ROUTER ----------------
if st.session_state.page == "home":
    home()
elif st.session_state.page == "faculty":
    faculty()
elif st.session_state.page == "student":
    student()
elif st.session_state.page == "samples":
    samples()