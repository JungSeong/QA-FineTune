"""
고양시 도서관 AI 사서 — Streamlit 챗봇
위치: QA-FineTune/main/web/app.py

실행 방법:
  # 1. server.py 먼저 실행
  uvicorn server:app --host 0.0.0.0 --port 9000

  # 2. Streamlit 실행
  streamlit run app.py
"""

import re
import requests
import streamlit as st

SERVER_URL = "http://localhost:9000"

MODEL_MODES = {
    "📚 Baseline  (파인튜닝 없음)":            "baseline",
    "📁 Just Excel  (엑셀 그대로 파인튜닝)":    "excel",
    "✨ SFT-LoRA  (DeepEval 데이터 파인튜닝)": "sft",
}


# ─────────────────────────────────────────────────────────
# API 호출
# ─────────────────────────────────────────────────────────

def get_status() -> dict:
    try:
        return requests.get(f"{SERVER_URL}/status", timeout=3).json()
    except Exception:
        return {"vllm": False, "rag": False, "current_model": "unknown"}


def query_model(mode: str, question: str, context: str, use_rag: bool, top_k: int) -> dict:
    try:
        resp = requests.post(
            f"{SERVER_URL}/query",
            json={"mode": mode, "question": question, "context": context, "use_rag": use_rag, "top_k": top_k},
            timeout=120,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("_x000D_", "\n")
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()


# ─────────────────────────────────────────────────────────
# UI 설정
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="고양시 도서관 AI 사서",
    page_icon="📖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2a2d3e;
    --accent:  #4a9eff;
    --text:    #e8eaf0;
    --muted:   #6b7280;
    --green:   #4ade80;
    --red:     #f87171;
    --yellow:  #fbbf24;
    --purple:  #a78bfa;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Noto Serif KR', serif;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.block-container { padding: 2rem 3rem !important; }

.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { color: var(--muted); font-size: 0.9rem; margin-top: 0.4rem; }

.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 11px; border-radius: 999px;
    font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
}
.pill-on  { background:#14532d33; color:var(--green);  border:1px solid #14532d; }
.pill-off { background:#7f1d1d33; color:var(--red);    border:1px solid #7f1d1d; }

/* RAG 토글 박스 */
.rag-box {
    background: #1a1d27;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
}
.rag-box-active {
    background: #0f1e3d;
    border: 1px solid #2a3a5e;
    border-left: 3px solid var(--purple);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
}
.rag-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--purple);
    margin-bottom: 0.3rem;
}
.rag-desc {
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.5;
}

.chat-user {
    background: #1e293b;
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.7rem 0 0.7rem 4rem;
    line-height: 1.7;
}
.chat-assistant {
    background: var(--surface);
    border: 1px solid #2a3a5e;
    border-left: 3px solid var(--accent);
    border-radius: 4px 12px 12px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.7rem 4rem 0.7rem 0;
    line-height: 1.8;
    white-space: pre-wrap;
}
.chat-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--muted);
    margin-top: 0.5rem; text-align: right;
}
.retrieved-doc {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    margin-top: 0.3rem;
    font-size: 0.8rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
}
.badge {
    display: inline-block; padding: 2px 9px;
    border-radius: 4px; font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500; margin-bottom: 0.4rem;
    margin-right: 4px;
}
.b-base   { background:#374151;    color:#9ca3af; }
.b-excel    { background:#1e3a5f;    color:var(--accent); }
.b-sft    { background:#14532d55;  color:var(--green); }
.b-rag    { background:#2e1a4a;    color:var(--purple); }

.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Noto Serif KR', serif !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px #4a9eff22 !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'Noto Serif KR', serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: var(--border) !important; }

/* 토글 색상 */
[data-testid="stCheckbox"] label {
    color: var(--text) !important;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── 헤더 ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📖 고양시 도서관 AI 사서</h1>
    <p>Baseline · Excel · SFT-LoRA 모델을 선택하여 도서관 FAQ를 질문해보세요</p>
</div>
""", unsafe_allow_html=True)


# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    st.divider()

    # 서버 상태
    status  = get_status()
    vllm_ok = status.get("vllm", False)
    rag_ok  = status.get("rag",  False)

    vllm_pill = f'<span class="pill {"pill-on" if vllm_ok else "pill-off"}">● vLLM {"연결됨" if vllm_ok else "오프라인"}</span>'
    rag_pill  = f'<span class="pill {"pill-on" if rag_ok  else "pill-off"}">● RAG {"연결됨"  if rag_ok  else "오프라인"}</span>'
    st.markdown(f"{vllm_pill}&nbsp;{rag_pill}", unsafe_allow_html=True)

    cur_model = status.get("current_model", "unknown")
    st.markdown(
        f'<div style="color:var(--muted);font-size:0.72rem;'
        f'font-family:\'JetBrains Mono\',monospace;margin-top:6px">'
        f'model: {cur_model}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # 모델 선택
    st.markdown("### 🤖 모델 선택")
    selected_label = st.radio("", list(MODEL_MODES.keys()), index=0)
    selected_mode  = MODEL_MODES[selected_label]

    st.divider()

    # context 입력 (Excel/SFT 전용)
    if selected_mode in ("excel", "sft"):
        st.markdown("### 📄 도서관 정보")
        context_input = st.text_area(
            "",
            placeholder="FAQ 원문이나 관련 도서관 규정을 입력하세요.\n비워두면 기본 안내 문구가 사용됩니다.",
            height=140,
            key="context",
        )
    else:
        context_input = ""
        st.info("Baseline: context 없이 질문만으로 답변합니다.")

    st.divider()

    # 심층적 사고 (RAG)
    st.markdown("### 🔎 심층적 사고")
    use_rag = st.toggle(
        "TOP_K 결정",
        value=False,
        disabled=bool(not rag_ok),  # ← bool()로 명시적 변환
    )

    if use_rag:
        top_k = st.slider("검색 문서 수", min_value=1, max_value=20, value=5, step=1)
    else:
        top_k = 5

    if use_rag and rag_ok:
        st.markdown("""
        <div class="rag-box-active">
            <div class="rag-label">🔍 RAG 활성화됨</div>
            <div class="rag-desc">
                질문 전송 시 pgvector에서 관련 문서를<br>
                자동으로 검색하여 답변에 반영합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not rag_ok:
        st.markdown("""
        <div class="rag-box">
            <div class="rag-desc" style="color:#f87171">
                RAG 검색 서버가 오프라인입니다.<br>
                search-data 컨테이너를 확인하세요.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rag-box">
            <div class="rag-desc">
                활성화 시 pgvector에서 관련 도서관<br>
                문서를 검색하여 답변 품질을 높입니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── 채팅 히스토리 ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_badges(mode: str, used_rag: bool) -> str:
    mode_badge = {
        "baseline": '<span class="badge b-base">Baseline</span>',
        "excel":      '<span class="badge b-excel">Excel</span>',
        "sft":      '<span class="badge b-sft">SFT-LoRA</span>',
    }.get(mode, "")
    rag_badge = '<span class="badge b-rag">+ RAG</span>' if used_rag else ""
    return mode_badge + rag_badge


for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user">🙋 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        badges  = get_badges(msg.get("mode", ""), msg.get("used_rag", False))
        elapsed = msg.get("elapsed", 0)
        docs    = msg.get("retrieved_docs", [])
        content = clean_text(msg["content"])

        doc_html = ""
        if docs:
            items = "".join(
                f'<div class="retrieved-doc">📄 {i+1}. '
                f'{d[:250].replace(chr(10), " ")}{"..." if len(d) > 250 else ""}</div>'
                for i, d in enumerate(docs)
            )
            doc_html = (
                f"<details><summary style='color:var(--muted);font-size:0.8rem;"
                f"cursor:pointer;margin-bottom:6px'>🔍 검색된 문서 {len(docs)}건</summary>"
                f"{items}</details>"
            )

        st.markdown(
            f'<div class="chat-assistant">'
            f'{badges}'
            f'{doc_html}'
            f'<div style="margin-top:0.4rem">{content}</div>'
            f'<div class="chat-meta">⏱ {elapsed:.2f}s</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── 입력창 ────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns([5, 1])

with col1:
    question = st.text_area(
        "질문",
        placeholder="예: 도서 대출 기간은 얼마나 되나요?",
        height=80,
        key="question_input",
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    send = st.button("전송 ↗", use_container_width=True)


# ── 추론 ──────────────────────────────────────────────────
if send and question.strip():
    if not vllm_ok:
        st.error("vLLM 서버에 연결할 수 없습니다.")
    elif use_rag and not rag_ok:
        st.error("RAG 검색 서버에 연결할 수 없습니다.")
    else:
        st.session_state.messages.append({
            "role":    "user",
            "content": question.strip(),
        })
        with st.spinner("사서가 답변을 작성 중입니다..."):
            result = query_model(
                selected_mode,
                question.strip(),
                context_input,
                use_rag,
                top_k
            )
            if "error" in result:
                st.error(f"오류: {result['error']}")
            else:
                st.session_state.messages.append({
                    "role":          "assistant",
                    "content":       result.get("answer", ""),
                    "mode":          selected_mode,
                    "elapsed":       result.get("elapsed", 0),
                    "used_rag":      use_rag,
                    "retrieved_docs": result.get("retrieved_docs", []),
                })
        st.rerun()

elif send and not question.strip():
    st.warning("질문을 입력해주세요.")