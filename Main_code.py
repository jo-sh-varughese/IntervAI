# Main_code.py
# IntervAI — Professional AI Interviewer (Dark theme, Plotly, PDF report)

import os
import re
import json
import threading
import platform
import tempfile
from typing import Dict, List

import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Plotly (interactive, dark theme) + static export via kaleido
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.chromium_args = ("--no-sandbox", "--disable-dev-shm-usage")


# Optional: voice input (works locally; typically disabled in cloud)
try:
    import speech_recognition as sr
except Exception:
    sr = None

# =========================
# App + Model Configuration
# =========================
PLOTLY_TEMPLATE = "plotly_dark"          # dark theme for all charts
MODEL_NAME = "gemini-2.5-flash"          # fast, widely available
MAX_RESUME_CHARS = 6000                   # safe token budget for resume context

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="IntervAI — Professional AI Interviewer",
    page_icon="🤖",
    layout="wide"
)

# Simple dark CSS touch-ups
st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; }
      .metric { text-align: center; }
      .badge { background:#1f2937; padding:6px 10px; border-radius:12px; margin-right:8px; font-size:12px; color:#e8eaed;}
      .section-title { font-size: 1.1rem; font-weight: 600; margin: 0.4rem 0 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Header
# =========================
colA, colB = st.columns([0.7, 0.3])
with colA:
    st.title("🤖 IntervAI — Professional AI Interviewer")
    st.caption("Dynamic, resume-aware interviewing • Real-time feedback • Interactive analytics • PDF reporting (dark mode)")
with colB:
    st.markdown(
        "<div style='text-align:right;'>"
        "<span class='badge'>Adaptive Q&A</span>"
        "<span class='badge'>Resume Intelligence</span>"
        "<span class='badge'>Scoring & Criteria</span>"
        "<span class='badge'>Plotly Analytics</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# =========================
# Sidebar (Controls)
# =========================
st.sidebar.header("🎯 Interview Settings")
role = st.sidebar.text_input("Job Role", "AI Engineer")
seniority = st.sidebar.selectbox("Seniority Level", ["Intern", "Junior", "Mid-level", "Senior", "Lead"])
personality = st.sidebar.selectbox("Interviewer Type", ["Technical", "HR", "Culture", "Managerial"])
jd = st.sidebar.text_area(
    "Job Description / Focus Areas",
    "Develop and deploy AI models. Requires Python, ML, NLP, and LLM experience."
)
temperature = st.sidebar.slider("AI Creativity", 0.0, 1.0, 0.3, 0.05)
max_questions = st.sidebar.slider("Max Questions", 3, 15, 7)

candidate_name = st.sidebar.text_input("Candidate Name (for PDF/report)", "John Doe")

uploaded_file = st.sidebar.file_uploader("📄 Upload Candidate Resume (PDF)", type=["pdf"])

# =========================
# Initialize Gemini
# =========================
if not API_KEY:
    st.error("❌ Add your Gemini API key in `.env` as `GOOGLE_API_KEY=...`")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Gemini init error: {e}")
    st.stop()

# =========================
# Resume Parsing & Summary
# =========================
resume_text = ""
if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        for p in reader.pages:
            resume_text += (p.extract_text() or "")
        st.sidebar.success("✅ Resume loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read PDF: {e}")

summary = ""
skills: List[str] = []
if resume_text.strip():
    skill_prompt = f"""
    Extract key info from this resume. Reply STRICT JSON with keys:
    {{
      "summary": "<2-3 sentence candidate summary>",
      "skills": ["skill1","skill2","skill3", "..."]
    }}
    Resume:
    {resume_text[:MAX_RESUME_CHARS]}
    """
    try:
        resp = model.generate_content(skill_prompt, generation_config={"temperature": 0.1})
        parsed = json.loads(resp.text)
        summary = parsed.get("summary", "")
        skills = parsed.get("skills", [])
    except Exception:
        summary = "Could not extract summary."
        skills = []
    with st.sidebar.expander("Resume Intelligence", expanded=True):
        st.markdown(f"**Summary:** {summary if summary else '—'}")
        st.markdown("**Skills:** " + (", ".join(skills) if skills else "—"))
else:
    st.sidebar.info("Upload a resume for deeper personalization (optional).")

# =========================
# Session State
# =========================
if "conversation" not in st.session_state:
    st.session_state.conversation: List[Dict] = []  # [{role, content}]
if "scores" not in st.session_state:
    st.session_state.scores: List[int] = []
if "criteria_list" not in st.session_state:
    st.session_state.criteria_list: List[Dict[str, float]] = []  # correctness/clarity/relevance/depth per Q
if "interview_done" not in st.session_state:
    st.session_state.interview_done = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# =========================
# Voice (Cloud-safe toggles)
# =========================
ENABLE_VOICE = False
VOICE_AVAILABLE = True
try:
    if "streamlit" in platform.node().lower() or os.environ.get("STREAMLIT_RUNTIME"):
        VOICE_AVAILABLE = False  # Streamlit Cloud: no TTS stack
    else:
        # Optional local TTS via pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)

            def speak(text: str):
                def _run():
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except RuntimeError:
                        pass
                threading.Thread(target=_run, daemon=True).start()
        except Exception:
            VOICE_AVAILABLE = False

    if VOICE_AVAILABLE:
        ENABLE_VOICE = st.sidebar.toggle("🔊 Voice output (local only)", value=False)
        if not ENABLE_VOICE:
            def speak(text: str): pass
    else:
        def speak(text: str): pass
        st.sidebar.info("🔇 Voice disabled in cloud/headless environments.")
except Exception:
    def speak(text: str): pass
    st.sidebar.info("🔇 Voice disabled.")

# Optional speech input (microphone usually unavailable on cloud)
VOICE_INPUT = False
if sr is not None and VOICE_AVAILABLE:
    VOICE_INPUT = st.sidebar.toggle("🎙️ Voice input (local)", value=False)

# =========================
# Prompt Policies
# =========================
SYSTEM_PROMPT = f"""
You are a professional {personality} interviewer for a {role} role.
You run realistic, structured interviews. You value clarity, reasoning, impact, and relevant experience.

Rules:
- Ask ONE concise question at a time (<50 words).
- Adapt difficulty to candidate performance.
- Use resume context (summary/skills) if present.
- Avoid repeats and illegal or biased questions.
- Keep tone friendly, professional, and to-the-point.
"""

def build_context() -> str:
    return (
        f"[ROLE] {role} ({seniority})\n"
        f"[JD] {jd}\n"
        f"[RESUME SUMMARY] {summary if summary else '(none)'}\n"
        f"[CANDIDATE SKILLS] {', '.join(skills) if skills else '(none)'}\n"
    )

# =========================
# Core LLM Helpers
# =========================
def next_question() -> str:
    context = build_context()
    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.conversation])
    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Conversation so far:
{history}

Write the next most relevant interview question based on the candidate's last answer.
Return ONLY the question text.
"""
    resp = model.generate_content(prompt, generation_config={"temperature": temperature})
    return resp.text.strip()

def evaluate_answer(question: str, answer: str) -> Dict:
    """
    Robust evaluator: returns {"score": int, "feedback": str, "criteria": {...}}
    Includes regex/guarded fallback if JSON parsing fails.
    """
    eval_prompt = f"""
You are evaluating a candidate's answer.

Question: {question}
Answer: {answer}

Respond STRICT JSON only:
{{
  "score": <integer 0-10>,
  "feedback": "<short constructive feedback>",
  "criteria": {{
    "correctness": <0-10>,
    "clarity": <0-10>,
    "relevance": <0-10>,
    "depth": <0-10>
  }}
}}
"""
    try:
        resp = model.generate_content(eval_prompt, generation_config={"temperature": 0.1})
        raw = (resp.text or "").strip()

        # Strict parse
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback parsing
            score = 5
            m = re.search(r'"?score"?\s*[:\-]\s*(\d+)', raw, re.I)
            if not m:
                m = re.search(r'(\d+)\s*/\s*10', raw)
            if m:
                try:
                    score = int(m.group(1))
                except Exception:
                    pass

            # Try to grab criteria
            criteria = {"correctness": 5, "clarity": 5, "relevance": 5, "depth": 5}
            for k in criteria.keys():
                mk = re.search(rf'"?{k}"?\s*[:\-]\s*(\d+)', raw, re.I)
                if mk:
                    try:
                        criteria[k] = int(mk.group(1))
                    except Exception:
                        pass

            # Feedback fallback
            fb = "Good start—add concrete examples, metrics, and trade-offs."
            mf = re.search(r'"?feedback"?\s*[:\-]\s*"(.*?)"', raw, re.I | re.S)
            if mf:
                fb = mf.group(1).strip()
            else:
                mf2 = re.search(r'feedback[:\-]\s*(.+)', raw, re.I)
                if mf2:
                    fb = mf2.group(1).strip()

            data = {"score": score, "feedback": fb, "criteria": criteria}

        # Bounds + types
        data["score"] = max(0, min(10, int(data.get("score", 5))))
        crit = data.get("criteria", {})
        criteria = {
            "correctness": max(0, min(10, int(crit.get("correctness", 5)))),
            "clarity":     max(0, min(10, int(crit.get("clarity", 5)))),
            "relevance":   max(0, min(10, int(crit.get("relevance", 5)))),
            "depth":       max(0, min(10, int(crit.get("depth", 5)))),
        }
        data["criteria"] = criteria
        if not data.get("feedback"):
            data["feedback"] = "Thanks. Add specifics (metrics, constraints, trade-offs)."
        return data

    except Exception as e:
        return {
            "score": 5,
            "feedback": f"Evaluation error: {e}",
            "criteria": {"correctness": 5, "clarity": 5, "relevance": 5, "depth": 5},
        }

# =========================
# UI: Top Controls & Chat
# =========================
top_controls = st.columns([1,1,2,2])
with top_controls[0]:
    if st.button("▶️ Start / Reset"):
        st.session_state.conversation = []
        st.session_state.scores = []
        st.session_state.criteria_list = []
        st.session_state.question_count = 0
        st.session_state.interview_done = False
        st.success("Session reset.")
with top_controls[1]:
    if st.button("Ask Next Question ▶️", disabled=st.session_state.interview_done):
        if st.session_state.question_count == 0:
            greet = f"I’m your {personality} interviewer for the {role} role. Let’s begin."
            st.session_state.conversation.append({"role": "assistant", "content": greet})
        q = next_question()
        st.session_state.conversation.append({"role": "assistant", "content": q})
        st.session_state.question_count += 1
        speak(q)

# Chat history
for m in st.session_state.conversation:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Input Row
input_cols = st.columns([4,1])
user_answer = input_cols[0].chat_input("Your answer…")
if VOICE_INPUT and sr is not None and input_cols[1].button("🎤 Speak"):
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening…")
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
        user_answer = r.recognize_google(audio)
        st.success(f"You said: {user_answer}")
    except Exception as e:
        st.warning(f"Voice input error: {e}")

if user_answer:
    # Store user answer
    st.session_state.conversation.append({"role": "user", "content": user_answer})

    # Evaluate
    question_text = ""
    for msg in reversed(st.session_state.conversation[:-1]):
        if msg["role"] == "assistant":
            question_text = msg["content"]
            break

    result = evaluate_answer(question_text, user_answer)
    st.session_state.scores.append(result["score"])
    st.session_state.criteria_list.append(result["criteria"])

    feedback_msg = f"**Feedback:** {result['feedback']}\n\n**Score:** {result['score']}/10"
    st.session_state.conversation.append({"role": "assistant", "content": feedback_msg})
    speak(result['feedback'])

    # Continue or finish
    if st.session_state.question_count < max_questions:
        nxt = next_question()
        st.session_state.conversation.append({"role": "assistant", "content": nxt})
        st.session_state.question_count += 1
        speak(nxt)
    else:
        st.session_state.conversation.append(
            {"role": "assistant", "content": "🎉 Thank you! We’ll share a summary shortly."}
        )
        st.session_state.interview_done = True

    st.rerun()

# =========================
# Analytics (Interactive Plotly)
# =========================
st.markdown("---")
st.subheader("📊 Interactive Analytics (Dark Mode)")

fig_line = None
fig_pie = None
fig_radar = None
avg_total = None
mean_crit = {}

if st.session_state.scores:
    # Line chart: score over time
    df_scores = pd.DataFrame({
        "Question #": list(range(1, len(st.session_state.scores) + 1)),
        "Score": st.session_state.scores
    })
    fig_line = px.line(
        df_scores,
        x="Question #",
        y="Score",
        markers=True,
        template=PLOTLY_TEMPLATE,
        title="Performance Over Time"
    )
    fig_line.update_yaxes(range=[0, 10])
    st.plotly_chart(fig_line, use_container_width=True)

    # Aggregate criteria (mean)
    crit_df = pd.DataFrame(st.session_state.criteria_list)
    mean_crit = crit_df.mean(numeric_only=True).to_dict()

    # Pie chart: strength composition
    pie_df = pd.DataFrame({
        "Criterion": list(mean_crit.keys()),
        "Average": list(mean_crit.values())
    })
    fig_pie = px.pie(
        pie_df,
        names="Criterion",
        values="Average",
        hole=0.45,
        template=PLOTLY_TEMPLATE,
        title="Overall Strength Composition"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Radar/polar
    fig_radar = go.Figure()
    cats = list(mean_crit.keys())
    vals = list(mean_crit.values())
    cats_close = cats + [cats[0]]
    vals_close = vals + [vals[0]]
    fig_radar.add_trace(go.Scatterpolar(r=vals_close, theta=cats_close, fill='toself', name='Averages'))
    fig_radar.update_layout(template=PLOTLY_TEMPLATE, title="Skill Radar", polar=dict(radialaxis=dict(range=[0,10])))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Summary metrics
    avg_total = round(df_scores["Score"].mean(), 1)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Average Score", f"{avg_total}/10")
    c2.metric("Correctness", f"{mean_crit.get('correctness', 0):.1f}/10")
    c3.metric("Clarity", f"{mean_crit.get('clarity', 0):.1f}/10")
    c4.metric("Relevance", f"{mean_crit.get('relevance', 0):.1f}/10")
    c5.metric("Depth", f"{mean_crit.get('depth', 0):.1f}/10")

# =========================
# PDF Report Generation
# =========================
st.markdown("---")
st.subheader("📄 Generate Professional PDF Report")

st.markdown(
    "<div class='section-title'>Includes: candidate name, role & level, interviewer type, "
    "average score, correctness, clarity, relevance, depth, and embedded charts.</div>",
    unsafe_allow_html=True,
)

def generate_pdf_report(name, role, seniority, personality, avg_total, mean_crit, figs):
    from fpdf import FPDF  # ensure imported in function scope for Streamlit

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "IntervAI Interview Report", ln=True, align="C")
    pdf.ln(8)

    # Candidate Info
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Candidate Name: {name}", ln=True)
    pdf.cell(0, 8, f"Role: {role} ({seniority})", ln=True)
    pdf.cell(0, 8, f"Interviewer Type: {personality}", ln=True)
    pdf.ln(5)

    # Average Metrics
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Overall Evaluation:", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Average Score: {avg_total}/10", ln=True)
    pdf.cell(0, 8, f"Correctness: {mean_crit.get('correctness',0):.1f}/10", ln=True)
    pdf.cell(0, 8, f"Clarity: {mean_crit.get('clarity',0):.1f}/10", ln=True)
    pdf.cell(0, 8, f"Relevance: {mean_crit.get('relevance',0):.1f}/10", ln=True)
    pdf.cell(0, 8, f"Depth: {mean_crit.get('depth',0):.1f}/10", ln=True)
    pdf.ln(8)

    # Save charts to temp files and embed
    temp_files = []
    for fig, title in figs:
        # Save chart as PNG
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.write_image(tmp.name)  # requires kaleido
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        pdf.image(tmp.name, w=180)
        pdf.ln(6)
        temp_files.append(tmp.name)

    # Footer
    pdf.set_font("Helvetica", "I", 10)
    pdf.ln(6)
    pdf.multi_cell(0, 6, "Generated by IntervAI — AI-powered interview assistant.\nhttps://github.com")

    # Output
    out_path = os.path.join(tempfile.gettempdir(), f"{name.replace(' ', '_')}_IntervAI_Report.pdf")
    pdf.output(out_path)
    return out_path, temp_files

if st.button("🧾 Generate PDF Report"):
    if not st.session_state.scores:
        st.warning("No data available. Complete an interview first.")
    else:
        # Rebuild figures if needed
        if avg_total is None or fig_line is None or fig_pie is None or fig_radar is None:
            # compute metrics
            df_scores = pd.DataFrame({"Question #": list(range(1, len(st.session_state.scores)+1)),
                                      "Score": st.session_state.scores})
            avg_total = round(df_scores["Score"].mean(), 1)
            crit_df = pd.DataFrame(st.session_state.criteria_list)
            mean_crit = crit_df.mean(numeric_only=True).to_dict()

            # charts
            fig_line = px.line(df_scores, x="Question #", y="Score", markers=True,
                               template=PLOTLY_TEMPLATE, title="Performance Over Time")
            fig_line.update_yaxes(range=[0,10])

            pie_df = pd.DataFrame({"Criterion": list(mean_crit.keys()), "Average": list(mean_crit.values())})
            fig_pie = px.pie(pie_df, names="Criterion", values="Average", hole=0.45,
                             template=PLOTLY_TEMPLATE, title="Overall Strength Composition")

            cats = list(mean_crit.keys())
            vals = list(mean_crit.values())
            cats_close = cats + [cats[0]]
            vals_close = vals + [vals[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=vals_close, theta=cats_close, fill='toself', name='Averages'))
            fig_radar.update_layout(template=PLOTLY_TEMPLATE, title="Skill Radar",
                                    polar=dict(radialaxis=dict(range=[0,10])))

        figs = [(fig_line, "Performance Over Time"),
                (fig_pie, "Overall Strength Composition"),
                (fig_radar, "Skill Radar")]

        pdf_path, tmp_files = generate_pdf_report(
            candidate_name, role, seniority, personality, avg_total, mean_crit, figs
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download Interview Report (PDF)",
                f,
                file_name=f"{candidate_name}_IntervAI_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        # Clean up temp images
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass

# =========================
# JSON/CSV Exports (optional)
# =========================
st.markdown("---")
st.subheader("📁 Export (CSV / JSON)")

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    if st.button("⬇️ Download CSV"):
        rows = []
        q_num = 0
        last_q = None
        for item in st.session_state.conversation:
            if item["role"] == "assistant" and not item["content"].startswith("**Feedback:**") and not item["content"].startswith("🎉"):
                last_q = item["content"]
            elif item["role"] == "user" and last_q:
                rows.append({"q_num": q_num+1, "question": last_q, "answer": item["content"]})
                q_num += 1
            elif item["role"] == "assistant" and item["content"].startswith("**Feedback:**") and rows:
                rows[-1]["feedback_block"] = item["content"]

        # attach scores/criteria
        for i, r in enumerate(rows):
            if i < len(st.session_state.scores):
                r["score"] = st.session_state.scores[i]
            if i < len(st.session_state.criteria_list):
                r.update(st.session_state.criteria_list[i])

        df_export = pd.DataFrame(rows)
        st.download_button(
            "Download interview_report.csv",
            df_export.to_csv(index=False).encode("utf-8"),
            "interview_report.csv",
            "text/csv",
            use_container_width=True
        )
with exp_col2:
    if st.button("⬇️ Download JSON"):
        export = {
            "candidate_name": candidate_name,
            "role": role,
            "seniority": seniority,
            "personality": personality,
            "jd": jd,
            "resume_summary": summary,
            "skills": skills,
            "scores": st.session_state.scores,
            "criteria_per_question": st.session_state.criteria_list,
            "conversation": st.session_state.conversation,
        }
        st.download_button(
            "Download interview_report.json",
            json.dumps(export, indent=2).encode("utf-8"),
            "interview_report.json",
            "application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.caption("IntervAI • Smart. Fair. Human. • Built with Streamlit, Gemini, Plotly & Kaleido (dark mode).")
