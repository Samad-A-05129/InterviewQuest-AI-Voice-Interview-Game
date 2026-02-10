import io
import re
import os
import csv
import audioop
import base64
import tempfile
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import streamlit as st
import speech_recognition as sr
import matplotlib.pyplot as plt
from gtts import gTTS
import requests

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ----------------------------
# MODERN UI INJECTOR (CSS)
# ----------------------------
def apply_modern_style():
    st.markdown("""
        <style>
        /* Main Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        /* Sidebar Polish */
        [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #eee; }
        
        /* Metric Card Styling */
        div[data-testid="stMetric"] {
            background: white;
            padding: 15px !important;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #f0f2f6;
        }
        
        /* Button Styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
            width: 100%;
        }
        .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        
        /* Question Card */
        .q-card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            border-left: 6px solid #4F46E5;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        
        /* Status Badges */
        .badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            background: #EEF2FF;
            color: #4338CA;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------------------------
# ORIGINAL FUNCTIONS (NO CHANGES)
# ----------------------------
def tts_play(text: str):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tmp_path = fp.name
    tts.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

st.set_page_config(page_title="InterviewQuest", page_icon="🎙️", layout="wide")
apply_modern_style()

# (The rest of your exact code follows below)

QUESTION_BANK = {
    "HR Interviewer": {
        1: [
            ("Tell me about yourself.", ["background", "skills", "experience", "strength", "goal"]),
            ("What are your strengths?", ["strength", "example", "impact", "team", "result"]),
            ("Why should we hire you?", ["fit", "skills", "value", "problem", "results"]),
        ],
        2: [
            ("Tell me about a time you handled a conflict.", ["situation", "action", "result", "communication", "resolution"]),
            ("Describe a challenge you overcame.", ["challenge", "task", "action", "result", "learning"]),
            ("Where do you see yourself in 3 years?", ["goal", "growth", "skills", "role", "learning"]),
        ],
        3: [
            ("Tell me about a failure and what you learned.", ["failure", "learning", "improve", "responsibility", "result"]),
            ("How do you handle pressure and deadlines?", ["pressure", "plan", "prioritize", "communicate", "deliver"]),
            ("Why this company?", ["company", "values", "role", "impact", "growth"]),
        ],
    },
    "Java / CS Interviewer": {
        1: [
            ("Explain OOP in simple words.", ["class", "object", "encapsulation", "inheritance", "polymorphism"]),
            ("Difference between compiler and interpreter?", ["compile", "interpret", "bytecode", "runtime", "errors"]),
            ("What is an array vs linked list?", ["array", "linked", "memory", "access", "insertion"]),
        ],
        2: [
            ("What is a database index and why is it used?", ["faster", "search", "lookup", "b-tree", "query"]),
            ("Difference between HTTP and HTTPS?", ["encryption", "ssl", "tls", "secure", "certificate"]),
            ("Explain stack vs queue with example.", ["stack", "queue", "lifo", "fifo", "example"]),
        ],
        3: [
            ("Explain time complexity with an example.", ["big o", "n", "log", "complexity", "example"]),
            ("What is multithreading? Give use case.", ["thread", "concurrency", "parallel", "race", "lock"]),
            ("Explain SOLID principles (any 2).", ["solid", "single", "open", "liskov", "dependency"]),
        ],
    },
    "ML Interviewer": {
        1: [
            ("What is Machine Learning in simple words?", ["data", "learn", "model", "predict", "patterns"]),
            ("Difference between supervised and unsupervised learning?", ["labels", "supervised", "unsupervised", "clustering", "classification"]),
            ("What is train-test split and why?", ["train", "test", "generalization", "overfitting", "validation"]),
        ],
        2: [
            ("What is precision vs recall?", ["precision", "recall", "false positive", "false negative", "tradeoff"]),
            ("Explain overfitting and how to reduce it.", ["overfitting", "generalization", "regularization", "dropout", "cross-validation"]),
            ("What is a confusion matrix?", ["tp", "tn", "fp", "fn", "matrix"]),
        ],
        3: [
            ("Explain how gradient descent works.", ["loss", "learning rate", "update", "minimize", "derivative"]),
            ("What is bias-variance tradeoff?", ["bias", "variance", "underfitting", "overfitting", "tradeoff"]),
            ("Explain ROC-AUC in simple terms.", ["roc", "auc", "threshold", "tpr", "fpr"]),
        ],
    },
}

BADGES = [
    (50, "Bronze Speaker 🥉"),
    (120, "Silver Speaker 🥈"),
    (220, "Gold Speaker 🥇"),
    (350, "Legend Voice 🎖️"),
]

LEADERBOARD_FILE = "leaderboard.csv"

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.title("⚙️ Settings")
role = st.sidebar.selectbox("Role / Interviewer Mode", list(QUESTION_BANK.keys()))
use_llm = st.sidebar.checkbox("Enable LLM Coach (Ollama)", value=True)
ollama_model = st.sidebar.text_input("Ollama model", value="llama3.2")
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
use_ai_questions = st.sidebar.checkbox("Use AI-generated questions (Ollama)", value=False)

game_type = st.sidebar.selectbox("Game Type", ["Single Player", "Multiplayer (2 players)"])
if game_type == "Single Player":
    player1 = st.sidebar.text_input("Player Name", value="Player 1")
    player2 = ""
else:
    player1 = st.sidebar.text_input("Player 1 Name", value="Player 1")
    player2 = st.sidebar.text_input("Player 2 Name", value="Player 2")

persist_leaderboard = st.sidebar.checkbox("Save Leaderboard to file", value=True)

client = None

# ----------------------------
# ORIGINAL LOGIC (NO CHANGES)
# ----------------------------
def ollama_chat(user_prompt: str, model: str, base_url: str) -> str:
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": False
        }
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"Ollama error: {e}"

def llm_feedback(question: str, answer: str) -> str:
    if not use_llm: return "LLM Coach is disabled."
    prompt = f"You are a strict but helpful interview coach.\n\nQuestion: {question}\nCandidate Answer: {answer}\n\nReturn in this format:\n\nStrengths:\n- ...\n\nMissing / Incorrect:\n- ...\n\nImprovements:\n- ...\n\nModel Answer:\n- ..."
    return ollama_chat(prompt, ollama_model, ollama_url)

def generate_ai_question(role_name: str, level_num: int):
    prompt = f"Generate ONE interview question for:\nRole: {role_name}\nDifficulty: {level_num}\nOnly return the question."
    return ollama_chat(prompt, ollama_model, ollama_url)

def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def keyword_score(answer: str, keywords: List[str]) -> Tuple[int, int]:
    ans = normalize_text(answer)
    hits = 0
    for k in keywords:
        if normalize_text(k) in ans: hits += 1
    return hits, len(keywords)

def structure_score(answer: str) -> int:
    ans = normalize_text(answer)
    length = len(ans.split())
    example_words = ["example", "for instance", "because", "so", "result", "impact", "therefore", "i did", "we did"]
    has_example = any(w in ans for w in example_words)
    s = 0
    if length >= 25: s += 25
    elif length >= 15: s += 15
    elif length >= 8: s += 8
    if has_example: s += 20
    return min(s, 45)

def clarity_score(answer: str) -> int:
    ans = normalize_text(answer)
    fillers = ["um", "uh", "like", "you know", "actually", "basically"]
    filler_count = sum(ans.count(f) for f in fillers)
    words = len(ans.split())
    if words == 0: return 0
    ratio = filler_count / max(words, 1)
    score = 30 - int(ratio * 120)
    return max(0, min(score, 30))

def final_score(answer: str, keywords: List[str]) -> Tuple[int, int, int, int, int, int]:
    struct_part = structure_score(answer)
    clarity_part = clarity_score(answer)
    if keywords and len(keywords) > 0:
        hits, total = keyword_score(answer, keywords)
        kw_part = int((hits / total) * 25) if total > 0 else 0
    else:
        hits, total = 0, 0
        kw_part = 25
    total_score = kw_part + struct_part + clarity_part
    return total_score, hits, total, kw_part, struct_part, clarity_part

def get_badge(points: int) -> Optional[str]:
    earned = None
    for p, name in BADGES:
        if points >= p: earned = name
    return earned

def analyze_audio_confidence(audio_data: sr.AudioData) -> Tuple[float, int, float]:
    raw = audio_data.get_raw_data()
    sample_rate = audio_data.sample_rate
    sample_width = audio_data.sample_width
    duration_sec = len(raw) / float(sample_rate * sample_width) if sample_rate and sample_width else 0.0
    try: rms = audioop.rms(raw, sample_width)
    except Exception: rms = 0
    chunk_ms = 30
    chunk_size = int(sample_rate * (chunk_ms / 1000.0) * sample_width)
    if chunk_size <= 0: return duration_sec, rms, 0.0
    thr = max(200, int(rms * 0.25))
    silent, total = 0, 0
    for i in range(0, len(raw), chunk_size):
        chunk = raw[i : i + chunk_size]
        if len(chunk) < chunk_size: break
        total += 1
        try: crms = audioop.rms(chunk, sample_width)
        except Exception: crms = thr
        if crms < thr: silent += 1
    silence_ratio = (silent / total) if total > 0 else 0.0
    return duration_sec, rms, silence_ratio

def listen_once(timeout: int = 5, phrase_time_limit: int = 15) -> Tuple[str, sr.AudioData, Optional[str]]:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.7)
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = r.recognize_google(audio)
        return text, audio, None
    except sr.UnknownValueError: return "", audio, "Sorry, I couldn't understand the audio."
    except sr.RequestError: return "", audio, "Speech service error (check internet)."
    except Exception as e: return "", audio, f"Mic error: {e}"

def load_leaderboard() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader: rows.append(r)
        except Exception: pass
    return rows

def save_leaderboard(rows: List[Dict[str, str]]):
    try:
        with open(LEADERBOARD_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "points", "date"])
            writer.writeheader()
            for r in rows: writer.writerow(r)
    except Exception: pass

def update_leaderboard(name: str, points: int) -> List[Dict[str, str]]:
    rows = load_leaderboard() if persist_leaderboard else st.session_state.get("leaderboard_rows", [])
    rows.append({"name": name, "points": str(points), "date": datetime.now().strftime("%Y-%m-%d %H:%M")})
    rows_sorted = sorted(rows, key=lambda x: int(x.get("points", "0")), reverse=True)[:20]
    if persist_leaderboard: save_leaderboard(rows_sorted)
    else: st.session_state.leaderboard_rows = rows_sorted
    return rows_sorted

def make_pdf_report(player_name: str, role_name: str, history: List[Dict[str, Any]], total_points: int) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "InterviewQuest - Performance Report")
    y -= 25
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Player: {player_name}"); y -= 15
    c.drawString(50, y, f"Role Mode: {role_name}"); y -= 15
    c.drawString(50, y, f"Total Points: {total_points}"); y -= 15
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"); y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Attempts (latest first):"); y -= 18
    c.setFont("Helvetica", 10)
    for h in reversed(history[-10:]):
        line1 = f"L{h['level']} | Score {h['score']} | Earned {h['earned']} | Q: {h['question']}"
        c.drawString(50, y, line1[:110]); y -= 12
        ans = (h.get("answer", "") or "").replace("\n", " ")
        c.drawString(60, y, ("Ans: " + ans)[:105]); y -= 16
        if y < 80:
            c.showPage(); y = height - 50; c.setFont("Helvetica", 10)
    c.showPage(); c.save(); buf.seek(0)
    return buf

# ----------------------------
# SESSION STATE
# ----------------------------
if "level" not in st.session_state: st.session_state.level = 1
if "points" not in st.session_state: st.session_state.points = 0
if "streak" not in st.session_state: st.session_state.streak = 0
if "q_index" not in st.session_state: st.session_state.q_index = 0
if "history" not in st.session_state: st.session_state.history = []
if "current_player" not in st.session_state: st.session_state.current_player = player1
if "p1_points" not in st.session_state: st.session_state.p1_points = 0
if "p2_points" not in st.session_state: st.session_state.p2_points = 0
if "current_qid" not in st.session_state: st.session_state.current_qid = ""
if "current_question" not in st.session_state: st.session_state.current_question = ""
if "current_keywords" not in st.session_state: st.session_state.current_keywords = []

if game_type == "Single Player": st.session_state.current_player = player1
else:
    if st.session_state.current_player not in [player1, player2]:
        st.session_state.current_player = player1

# ----------------------------
# HEADER UI
# ----------------------------
st.title("🎙️ InterviewQuest")
st.caption("AI Voice Interview Game • Speak your way to success")

# Top Stats Bar
with st.container():
    if game_type == "Single Player":
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", role)
        c2.metric("Level", st.session_state.level)
        c3.metric("Total Points", st.session_state.points)
        badge = get_badge(st.session_state.points)
        if badge: st.success(f"🏅 Badge unlocked: **{badge}**")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Turn", st.session_state.current_player)
        c2.metric(f"{player1}", st.session_state.p1_points)
        c3.metric(f"{player2}", st.session_state.p2_points)

st.divider()

# Game Settings
with st.expander("🎮 Game Settings & Modes", expanded=False):
    mode = st.selectbox("Choose mode", ["Normal", "Speed Round (10s)", "Boss Round (x2 points)"])
    phrase_limit = 10 if mode == "Speed Round (10s)" else 15
    hint = st.checkbox("Need a hint (show 1 keyword)?")

# ----------------------------
# QUESTION LOGIC
# ----------------------------
level = st.session_state.level
role_questions = QUESTION_BANK[role]
q_list = role_questions.get(level, role_questions[1])
q_index = st.session_state.q_index % len(q_list)

qid = f"{role}|L{level}|Q{st.session_state.q_index}|AI{int(use_ai_questions)}"
if st.session_state.current_qid != qid:
    q_bank, k_bank = q_list[q_index]
    if use_ai_questions:
        ai_q = generate_ai_question(role, level)
        if ai_q:
            st.session_state.current_question = ai_q
            st.session_state.current_keywords = []
        else:
            st.session_state.current_question = q_bank
            st.session_state.current_keywords = k_bank
    else:
        st.session_state.current_question = q_bank
        st.session_state.current_keywords = k_bank
    st.session_state.current_qid = qid

question = st.session_state.current_question
keywords = st.session_state.current_keywords

# DISPLAY QUESTION
st.markdown(f"""<div class="q-card">
    <p style="margin:0; font-weight:800; color:#4F46E5; font-size:0.8rem; text-transform:uppercase;">Question Level {level}</p>
    <h2 style="margin:10px 0; border:none;">{question}</h2>
</div>""", unsafe_allow_html=True)

if st.button("🔊 AI Ask Question (Speak)"):
    tts_play(f"Next question. {question}")

if hint:
    if keywords and len(keywords) > 0: st.info(f"Hint keyword: **{keywords[0]}**")
    else: st.info("Hint: AI-generated question. Focus on STAR method.")

# ----------------------------
# INPUT AREA
# ----------------------------
st.divider()
colL, colR = st.columns(2)
start_listen = colL.button("🎤 Start Listening", use_container_width=True)
manual = colR.text_input("Or type your answer (backup):")

answer_text = ""
audio_blob = None

if start_listen:
    with st.spinner("Listening... Speak now"):
        text, audio, err = listen_once(timeout=5, phrase_time_limit=phrase_limit)
    audio_blob = audio
    if err: st.warning(err)
    else:
        answer_text = text
        st.success("Captured!")

if manual.strip(): answer_text = manual.strip()

# ----------------------------
# EVALUATION (NO LOGIC CHANGES)
# ----------------------------
if answer_text:
    st.subheader("✅ Your Answer (Text)")
    st.write(answer_text)
    score, hits, total, kw_part, struct_part, clarity_part = final_score(answer_text, keywords)
    earned = score
    if mode == "Boss Round (x2 points)": earned *= 2
    
    if game_type == "Single Player":
        if score >= 60: st.session_state.streak += 1
        else: st.session_state.streak = 0
        st.session_state.points += earned
        if st.session_state.points >= 100 and st.session_state.level == 1:
            st.session_state.level = 2; st.balloons(); st.success("Level Up! Lv 2 🔥")
        if st.session_state.points >= 220 and st.session_state.level == 2:
            st.session_state.level = 3; st.balloons(); st.success("Level Up! Lv 3 🚀")
    else:
        if st.session_state.current_player == player1: st.session_state.p1_points += earned
        else: st.session_state.p2_points += earned

    # Analysis Layout
    a1, a2 = st.columns(2)
    with a1:
        st.subheader("🔊 Voice Analysis")
        if audio_blob is not None:
            dur, rms, silence_ratio = analyze_audio_confidence(audio_blob)
            words = len(normalize_text(answer_text).split())
            wps = (words / dur) if dur > 0 else 0
            st.write(f"- Duration: **{dur:.2f}s** | RMS: **{rms}**")
            st.write(f"- Speaking rate: **{wps:.2f} wps**")
            if rms < 800: st.write("• Speak louder.")
            if wps > 3.2: st.write("• Slow down.")
            if silence_ratio > 0.45: st.write("• Smoother flow needed.")
        else: st.write("Use mic for voice data.")

    with a2:
        st.subheader("📊 NLP Score Breakdown")
        st.write(f"**Total:** {score}/100 | **Earned:** {earned}")
        st.progress(min(score, 100) / 100)
        st.write(f"Keywords: {kw_part} | Struct: {struct_part} | Clarity: {clarity_part}")

    st.subheader("🤖 AI Coach Feedback")
    with st.spinner("Analyzing..."): coach_text = llm_feedback(question, answer_text)
    st.info(coach_text)

    st.session_state.history.append({
        "role": role, "level": level, "question": question, "answer": answer_text,
        "score": score, "earned": earned, "mode": mode, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ai_question": bool(use_ai_questions and (not keywords)),
    })

    if st.button("➡️ Next Question", type="primary"):
        st.session_state.q_index += 1
        st.session_state.current_qid = ""
        if game_type != "Single Player" and player2.strip():
            st.session_state.current_player = player2 if st.session_state.current_player == player1 else player1
        st.rerun()

# ----------------------------
# FOOTER / DASHBOARD
# ----------------------------
st.divider()
tab1, tab2, tab3 = st.tabs(["📜 History", "📈 Analytics", "🏆 Leaderboard"])

with tab1:
    if not st.session_state.history: st.write("No attempts.")
    else:
        for i, h in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.write(f"**{h['time']} - Score: {h['score']}**")
            st.write(f"Q: {h['question']}")
            st.divider()

with tab2:
    if st.session_state.history:
        scores = [h["score"] for h in st.session_state.history]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(scores, marker='o', color='#4F46E5')
        ax.set_title("Score Improvement")
        st.pyplot(fig)
        
        pdf_buf = make_pdf_report(player1 if game_type=="Single Player" else st.session_state.current_player, role, st.session_state.history, st.session_state.points)
        st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="report.pdf", mime="application/pdf")
    else: st.write("No data.")

with tab3:
    if st.button("📌 Submit Score"):
        if game_type == "Single Player": update_leaderboard(player1 or "Player", st.session_state.points)
        else: 
            update_leaderboard(player1 or "P1", st.session_state.p1_points)
            update_leaderboard(player2 or "P2", st.session_state.p2_points)
        st.success("Submitted!")
    
    rows = load_leaderboard() if persist_leaderboard else st.session_state.get("leaderboard_rows", [])
    if rows:
        for i, r in enumerate(sorted(rows, key=lambda x: int(x.get("points", "0")), reverse=True)[:10], 1):
            st.write(f"**{i}. {r['name']}** — {r['points']} pts")

if st.button("🔁 Reset Game"):
    st.session_state.clear()
    st.rerun()