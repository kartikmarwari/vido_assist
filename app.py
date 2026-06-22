import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_process import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title          # ← fixed import
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()
# Bridge Streamlit secrets → env vars (works locally via .env, on cloud via secrets)
import streamlit as st

if "PROXY_URL" in st.secrets:
    os.environ["PROXY_URL"] = st.secrets["PROXY_URL"]
if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
 
# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --surface-3: #20202e;
    --border: #2a2a3a;
    --border-light: #34344a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #8888a8;
    --text-faint: #5c5c80;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 14px;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(circle at 15% 10%, rgba(124,58,237,0.10), transparent 45%),
        radial-gradient(circle at 85% 90%, rgba(6,182,212,0.08), transparent 45%),
        linear-gradient(rgba(124,58,237,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.025) 1px, transparent 1px);
    background-size: auto, auto, 40px 40px, 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.9rem, 4.2vw, 3rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 55%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
    height: 100%;
}
.card:hover { border-color: var(--border-light); }
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-content {
    font-size: 0.88rem;
    line-height: 1.75;
    color: var(--text);
    white-space: pre-wrap;
}
.card-content.empty {
    color: var(--text-faint);
    font-style: italic;
    font-size: 0.82rem;
}

.badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 5px;
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.18); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.14);  color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.14); color: var(--success);     border: 1px solid rgba(16,185,129,0.3); }
.badge-amber  { background: rgba(245,158,11,0.14); color: var(--warning);     border: 1px solid rgba(245,158,11,0.3); }

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-faint) !important; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
    border: 1px dashed var(--border-light) !important;
    border-radius: 9px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

.stRadio [role="radiogroup"] { gap: 0.4rem; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.18s !important;
    text-transform: uppercase !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button:disabled {
    opacity: 0.45 !important;
    transform: none !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
}
.stDownloadButton > button {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text) !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    width: 100%;
}
.stDownloadButton > button:hover { border-color: var(--accent-2) !important; }

.status-bar {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.62rem 0.9rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.35rem 0;
    border: 1px solid var(--border);
    font-size: 0.76rem;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-active  { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.4s infinite; }
.dot-done    { background: var(--success); }
.dot-pending { background: var(--border-light); }
.dot-error   { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] {
    background: transparent !important;
    border-top: 1px solid var(--border) !important;
}

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    font-size: 0.8rem;
    line-height: 1.85;
    max-height: 320px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label, .stRadio label p { color: var(--text-muted) !important; font-size: 0.8rem !important; }

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
    .card { padding: 1.1rem; }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "pipeline_error": None,
    "uploaded_temp_path": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

PIPELINE_STEPS = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️", "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
]

STEP_HINTS = {
    "audio":      "Downloading / converting your audio…",
    "transcript": "Slowest step — can take a few minutes for long recordings.",
    "title":      "Almost there…",
    "summary":    "Reading through the full transcript…",
    "extract":    "Pulling out action items, decisions, and open questions…",
    "rag":        "Setting up chat so you can ask questions…",
}

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_css(steps, key):
    s = steps.get(key, "pending")
    return {"active": "dot-active", "done": "dot-done", "error": "dot-error"}.get(s, "dot-pending")

def render_step_bar(label, key, icon):
    css = step_css(st.session_state.pipeline_steps, key)
    st.markdown(
        f'<div class="status-bar"><div class="status-dot {css}"></div><span>{icon} {label}</span></div>',
        unsafe_allow_html=True,
    )

def card(title, content, empty_msg="Nothing found."):
    is_empty = not content or not str(content).strip()
    cls = "card-content empty" if is_empty else "card-content"
    shown = empty_msg if is_empty else content
    st.markdown(
        f'<div class="card"><div class="card-title">{title}</div>'
        f'<div class="{cls}">{shown}</div></div>',
        unsafe_allow_html=True,
    )

def reset_pipeline_state():
    st.session_state.pipeline_done = False
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.pipeline_steps = {}
    st.session_state.pipeline_error = None

def render_live_progress(container, steps_order, current_steps):
    total = len(steps_order)
    done_count = sum(1 for k, _, _ in steps_order if current_steps.get(k) == "done")
    with container.container():
        st.markdown(
            '<div class="card" style="margin-bottom:0.75rem">'
            '<div class="card-title">⚙️ Processing your file — please wait</div>',
            unsafe_allow_html=True,
        )
        st.progress(done_count / total if total else 0)
        rows_html = ""
        for key, icon, label in steps_order:
            state = current_steps.get(key, "pending")
            if state == "done":
                marker, style = "✅", "color:var(--success)"
            elif state == "active":
                marker, style = "🔄", "color:var(--accent-glow);font-weight:600"
            elif state == "error":
                marker, style = "❌", "color:var(--danger)"
            else:
                marker, style = "⏳", "color:var(--text-faint)"
            hint = (
                f' <span style="color:var(--text-faint);font-size:0.78rem">— {STEP_HINTS.get(key,"")}</span>'
                if state == "active" else ""
            )
            rows_html += (
                f'<div style="padding:0.45rem 0;font-size:0.85rem;{style}">'
                f'{marker} {icon} {label}{hint}</div>'
            )
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:var(--text-faint);font-size:0.72rem;line-height:1.5;margin:0.4rem 0 0.7rem">'
        'Works with meeting recordings, Zoom/Meet/Teams calls, interviews, podcasts, '
        'or any spoken-word video/audio.</div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Source type",
        ["YouTube URL", "Upload file", "Local file path"],
        label_visibility="collapsed",
    )

    source = None
    uploaded_file = None

    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube URL", placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        st.caption("Any public YouTube video — meetings, webinars, interviews, lectures.")
    elif input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload audio/video", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"],
            label_visibility="collapsed",
        )
        st.caption("Upload a Meet/Zoom/Teams recording, voice memo, or any audio/video file.")
    else:
        source = st.text_input(
            "Local file path", placeholder="/path/to/file.mp4",
            label_visibility="collapsed",
        )
        st.caption("Full path to a file already on this machine.")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done or any(st.session_state.pipeline_steps.values()):
        st.markdown("---")
        if st.session_state.pipeline_error:
            st.markdown('<span class="badge badge-amber">Pipeline Status</span>', unsafe_allow_html=True)
        elif st.session_state.pipeline_done:
            st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-cyan">Pipeline Status</span>', unsafe_allow_html=True)

        for step, icon, label in PIPELINE_STEPS:
            render_step_bar(label, step, icon)

        if st.session_state.pipeline_error:
            st.error(st.session_state.pipeline_error, icon="⚠️")

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">YouTube · Uploaded Recordings · Local Files — Transcribe, Summarise, Chat</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── Run Pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    resolved_source = None
    validation_error = None

    if input_mode == "Upload file":
        if uploaded_file is None:
            validation_error = "Please upload a file first."
        else:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".tmp"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.getvalue())
            tmp.close()
            resolved_source = tmp.name
            st.session_state.uploaded_temp_path = tmp.name
    else:
        if not source or not source.strip():
            placeholder = "YouTube URL" if input_mode == "YouTube URL" else "file path"
            validation_error = f"Please enter a {placeholder}."
        elif input_mode == "YouTube URL" and not source.strip().startswith(("http://", "https://")):
            validation_error = "Doesn't look like a valid URL — it should start with http:// or https://"
        elif input_mode == "Local file path" and not os.path.exists(source.strip()):
            validation_error = f"File not found: {source.strip()}"
        else:
            resolved_source = source.strip()

    if validation_error:
        st.error(validation_error, icon="🚫")
    else:
        reset_pipeline_state()
        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            render_live_progress(progress_placeholder, PIPELINE_STEPS, st.session_state.pipeline_steps)

        try:
            render_live_progress(progress_placeholder, PIPELINE_STEPS, st.session_state.pipeline_steps)

            # ── Audio + Transcript ───────────────────────────────────────────
            update_step("audio", "active")

            if input_mode == "YouTube URL":
                try:
                    # First: try audio download + Whisper
                    chunks = process_input(resolved_source)
                    update_step("audio", "done")
                    update_step("transcript", "active")
                    transcript = transcribe_all(chunks, language)
                    update_step("transcript", "done")
                except Exception:
                    # Fallback: fetch YouTube captions directly
                    try:
                        from youtube_transcript_api import YouTubeTranscriptApi
                        import re
                        vid = re.search(
                            r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", resolved_source
                        ).group(1)
                        proxy = os.getenv("PROXY_URL")
                        
                        if proxy:
                            from youtube_transcript_api.proxies import GenericProxyConfig
                            ytt_api = YouTubeTranscriptApi(
                                proxy_config=GenericProxyConfig(
                                http_url=proxy,
                                https_url=proxy,
        )
    )
                        else:
                             ytt_api = YouTubeTranscriptApi()
                        fetched = ytt_api.fetch(vid)
                        transcript = " ".join(s.text for s in fetched)
                        
                        

                        update_step("audio", "done")
                        update_step("transcript", "done")
                    except Exception as e:
                        raise Exception(f"Both audio download and transcript extraction failed: {e}")
            else:
                # File upload / local path — standard pipeline
                chunks = process_input(resolved_source)
                update_step("audio", "done")
                update_step("transcript", "active")
                transcript = transcribe_all(chunks, language)
                update_step("transcript", "done")

            # ── Rest of pipeline ─────────────────────────────────────────────
            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            render_live_progress(progress_placeholder, PIPELINE_STEPS, st.session_state.pipeline_steps)
            progress_placeholder.success("✅ Analysis complete!")
            st.rerun()

        except Exception as e:
            for k, _, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "error"
                    break
            st.session_state.pipeline_error = str(e)
            progress_placeholder.error(f"❌ {e}")
# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        card("📋 Summary", r["summary"], "No summary generated.")
    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇ Download transcript (.txt)",
                data=r["transcript"],
                file_name=f"{r['title'][:50] or 'transcript'}.txt",
                use_container_width=True,
            )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        card("✅ Action Items", r["action_items"], "No action items found.")
    with c2:
        card("🔑 Key Decisions", r["key_decisions"], "No key decisions found.")
    with c3:
        card("❓ Open Questions", r["open_questions"], "No open questions found.")

    st.markdown("---")

    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:1.15rem;font-weight:700;margin-bottom:0.75rem">'
        '💬 Chat with your Meeting</div>',
        unsafe_allow_html=True,
    )

    chat_box = st.container(height=380, border=True)
    with chat_box:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="text-align:center;color:var(--text-faint);padding:2rem 1rem;font-size:0.85rem">'
                '💬 Ask anything about your meeting — try "What were the main decisions?"'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                avatar = "🧑" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])

    user_input = st.chat_input("What were the main decisions made?")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking…"):
            try:
                answer = ask_question(r["rag_chain"], user_input)
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        cols = st.columns([4, 1])
        with cols[1]:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem 1.5rem">
        <div style="font-size:3.5rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Works with any spoken-word recording
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:480px;line-height:1.7;margin:0 auto">
            Drop in a meeting recording, a call export, an interview, or any video link —
            pick your language in the sidebar, and hit <strong>Analyse</strong>.
        </div>
    </div>""", unsafe_allow_html=True)

    uc1, uc2, uc3 = st.columns(3, gap="medium")
    with uc1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📹 YouTube URL</div>
            <div class="card-content" style="font-size:0.8rem">
                Webinars, lectures, interviews, conference talks — paste any public link.
            </div>
        </div>""", unsafe_allow_html=True)
    with uc2:
        st.markdown("""
        <div class="card">
            <div class="card-title">⬆️ Upload a Recording</div>
            <div class="card-content" style="font-size:0.8rem">
                Google Meet / Zoom / Teams exports, voice memos, podcasts — mp3, wav, m4a, mp4, mov, mkv.
            </div>
        </div>""", unsafe_allow_html=True)
    with uc3:
        st.markdown("""
        <div class="card">
            <div class="card-title">💻 Local File Path</div>
            <div class="card-content" style="font-size:0.8rem">
                Already have the file on this machine? Point to its full path.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;color:var(--text-faint);font-size:0.78rem;margin:1.5rem 0 0.75rem">'
        'WHAT YOU GET BACK</div>',
        unsafe_allow_html=True,
    )

    oc1, oc2, oc3, oc4 = st.columns(4, gap="small")
    for col, icon, label in [
        (oc1, "📝", "Full transcript"),
        (oc2, "✅", "Action items"),
        (oc3, "🔑", "Key decisions"),
        (oc4, "❓", "Open questions"),
    ]:
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:0.9rem 0.5rem;background:var(--surface);'
                f'border:1px solid var(--border);border-radius:10px;font-size:0.8rem">'
                f'<div style="font-size:1.4rem;margin-bottom:0.3rem">{icon}</div>{label}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="text-align:center;margin-top:1.5rem">'
        '<span class="badge badge-purple">Transcription</span>&nbsp;'
        '<span class="badge badge-cyan">Summarisation</span>&nbsp;'
        '<span class="badge badge-green">RAG Chat</span>'
        '</div>',
        unsafe_allow_html=True,
    )
