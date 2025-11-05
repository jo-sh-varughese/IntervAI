import os
import json
import threading
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import speech_recognition as sr
from fpdf import FPDF
import matplotlib.pyplot as plt
import platform

# ------------------ Load API Key ------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ No Gemini API key found. Add it in a `.env` file as GOOGLE_API_KEY=your_key_here")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="IntervAI – AI Interviewer", page_icon="🤖", layout="wide")
st.title("🤖 IntervAI — The Intelligent Job Interviewer")
st.caption("Adaptive, voice-ready AI interviews powered by Google Gemini, Streamlit, and Python.")

# ------------------ Sidebar Configuration ------------------
st.sidebar.header("🎯 Interview Configuration")
role = st.sidebar.text_input("Job Role", "AI Engineer")
seniority = st.sidebar.selectbox("Seniority Level", ["Intern", "Junior", "Mid-level", "Senior", "Lead"])
personality = st.sidebar.selectbox("Interviewer Type", ["Technical", "HR", "Culture"])
jd = st.sidebar.text_area(
    "Job Description / Focus Areas",
    "Responsible for developing and deploying AI models. Requires Python, ML, NLP, and LLM experience."
)
temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.4, 0.05)
max_questions = st.sidebar.slider("Max Questions", 3, 10, 5)

# ------------------ Resume Upload ------------------
uploaded_file = st.sidebar.file_uploader("📄 Upload Candidate Resume (PDF)", type=["pdf"])
resume_text = ""
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        resume_text += page.extract_text()
    st.sidebar.success("✅ Resume uploaded successfully!")
else:
    st.sidebar.info("Upload a PDF resume to personalize interview questions.")

# ------------------ Session State ------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "scores" not in st.session_state:
    st.session_state.scores = []
if "interview_done" not in st.session_state:
    st.session_state.interview_done = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ------------------ Voice Setup (Cloud-Safe) ------------------
ENABLE_VOICE = True
try:
    # Disable voice in Streamlit Cloud or limited environments
    if "streamlit" in platform.node().lower() or os.environ.get("STREAMLIT_RUNTIME"):
        ENABLE_VOICE = False
        st.sidebar.info("🔇 Voice output disabled (cloud environment).")
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)

        def speak(text):
            """Run TTS safely in a separate thread."""
            def _speak():
                try:
                    engine.say(text)
                    engine.runAndWait()
                except RuntimeError:
                    pass
            threading.Thread(target=_speak, daemon=True).start()
except Exception as e:
    ENABLE_VOICE = False
    st.sidebar.warning(f"Voice disabled: {e}")

# Fallback for cloud
if not ENABLE_VOICE:
    def speak(text):
        pass

# ------------------ Speech Recognition ------------------
r = sr.Recognizer()

def listen():
    """Capture voice answer."""
    with sr.Microphone() as source:
        st.info("🎙️ Listening... please speak your answer")
        audio = r.listen(source, timeout=5, phrase_time_limit=30)
    try:
        text = r.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("⚠️ Could not understand your speech.")
        return ""
    except sr.RequestError:
        st.error("⚠️ Speech recognition unavailable.")
        return ""

# ------------------ System Prompt ------------------
SYSTEM_PROMPT = f"""
You are an expert {personality} interviewer for AI and technical roles.
Conduct a realistic, adaptive interview.

Rules:
1. Ask one question at a time.
2. Use the candidate's resume (if given) for personalized questions.
3. Adjust difficulty dynamically.
4. Be concise, insightful, and professional.
5. Never repeat questions.
"""

# ------------------ Core Logic ------------------
def generate_question(role, seniority, jd, history, resume_text):
    """Generate the next dynamic interview question."""
    context = f"Role: {role} ({seniority}). JD: {jd}. Resume: {resume_text[:1500]}"
    conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    prompt = f"""{SYSTEM_PROMPT}

Context: {context}

Conversation so far:
{conversation_text}

Ask the next most relevant interview question based on the candidate's last answer.
Return only the question text.
"""
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text.strip()

def evaluate_answer(question, answer, role, seniority, jd):
    """Evaluate candidate’s answer and provide feedback."""
    prompt = f"""
You are evaluating a candidate's answer for a {role} ({seniority}) position.
Job Description: {jd}

Question: {question}
Answer: {answer}

Rate from 0-10 for correctness, clarity, and relevance.
Return JSON: {{"score": <number>, "feedback": "<short feedback>"}}.
"""
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    try:
        result = json.loads(response.text.strip())
    except Exception:
        result = {"score": 5, "feedback": "Could not parse evaluation."}
    return result

# ------------------ Main Interview Flow ------------------
if not st.session_state.interview_done:
    if st.session_state.question_count == 0:
        if st.button("Start Interview ▶️"):
            first_q = generate_question(role, seniority, jd, [], resume_text)
            st.session_state.conversation.append({"role": "assistant", "content": first_q})
            st.session_state.question_count += 1
            speak(first_q)
            st.rerun()
    else:
        # Display conversation
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

        col1, col2 = st.columns([2, 1])
        with col1:
            user_input = st.chat_input("Your answer (or click 🎤 to speak):")
        with col2:
            if st.button("🎤 Speak"):
                user_input = listen()

        if user_input:
            st.session_state.conversation.append({"role": "user", "content": user_input})
            question = st.session_state.conversation[-2]["content"]

            evaluation = evaluate_answer(question, user_input, role, seniority, jd)
            feedback = f"**Feedback:** {evaluation['feedback']}\n**Score:** {evaluation['score']}/10"
            st.session_state.scores.append(evaluation['score'])
            st.session_state.conversation.append({"role": "assistant", "content": feedback})
            speak(evaluation['feedback'])

            if st.session_state.question_count < max_questions:
                next_q = generate_question(role, seniority, jd, st.session_state.conversation, resume_text)
                st.session_state.conversation.append({"role": "assistant", "content": next_q})
                st.session_state.question_count += 1
                speak(next_q)
            else:
                st.session_state.conversation.append(
                    {"role": "assistant", "content": "🎉 Thank you for the interview! We'll get back to you soon."}
                )
                st.session_state.interview_done = True
            st.rerun()

else:
    st.success("✅ Interview completed!")

    # 📊 Dashboard
    if st.session_state.scores:
        avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
        st.metric("Average Score", f"{avg_score:.1f}/10")

        fig, ax = plt.subplots()
        ax.plot(range(1, len(st.session_state.scores)+1), st.session_state.scores, marker='o')
        ax.set_title("Candidate Performance Progress")
        ax.set_xlabel("Question #")
        ax.set_ylabel("Score")
        st.pyplot(fig)

    # 🗂️ Export Options
    export_choice = st.radio("📤 Export Report As:", ["None", "CSV", "PDF"])
    if export_choice == "CSV":
        df = pd.DataFrame(st.session_state.conversation)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "interview_report.csv", "text/csv")
    elif export_choice == "PDF":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="AI Interview Report", ln=True, align="C")
        for msg in st.session_state.conversation:
            role = msg['role'].capitalize()
            content = msg['content']
            pdf.multi_cell(0, 10, f"{role}: {content}\n")
        pdf.output("interview_report.pdf")
        with open("interview_report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "interview_report.pdf", "application/pdf")

    if st.button("Start New Interview 🔄"):
        st.session_state.conversation = []
        st.session_state.scores = []
        st.session_state.question_count = 0
        st.session_state.interview_done = False
        st.rerun()
