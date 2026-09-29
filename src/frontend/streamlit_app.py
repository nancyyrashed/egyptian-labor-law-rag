"""
Phase 4: Streamlit frontend (bilingual: Arabic / English).

Calls the FastAPI backend (main.py) over HTTP - a separate process from
the API, matching the planned architecture (FastAPI + Streamlit).

Run (in a separate terminal from the FastAPI server):

    streamlit run src/frontend/streamlit_app.py

Requires the FastAPI server already running:

    uvicorn src.api.main:app --reload
"""

# Some hosts (notably Streamlit Community Cloud) ship an old system SQLite
# that Chroma may reject. If pysqlite3-binary is installed, use its newer
# SQLite instead. Harmless no-op everywhere else (e.g. Windows, no package).
try:
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import html
import os
import sys

import requests
import streamlit as st


# Overridable so the same code runs locally (default) and inside the
# Docker image, where the API is reachable at 127.0.0.1 inside the container.
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")

# RAG_MODE selects HOW questions are answered:
#   "api"    (default) - call the separate FastAPI backend at API_URL.
#                        Used locally and in the Docker image.
#   "direct"           - run the pipeline (ask.py) inside this Streamlit
#                        process, no separate API. Used on Streamlit
#                        Community Cloud, which only runs the Streamlit app.
# On Streamlit Community Cloud it is set in the app's Secrets. Secrets are
# not visible at import time on every host, so it is resolved lazily below.
def _rag_mode() -> str:
    mode = os.getenv("RAG_MODE")
    if not mode:
        try:
            mode = st.secrets.get("RAG_MODE")
        except Exception:
            mode = None
    return (mode or "api").strip().lower()


RAG_MODE = _rag_mode()


# ---------------------------------------------------------------- texts
# Every user-visible string lives here so both languages stay in sync.

TEXT = {
    "ar": {
        "dir": "rtl",
        "page_title": "مساعد معلومات قانون العمل المصري",
        "title": "مساعد معلومات قانون العمل المصري",
        "subtitle": "قانون العمل رقم 14 لسنة 2025",
        "disclaimer": (
            "هذا النظام يقدم معلومات قانونية عامة استنادًا إلى قانون العمل المصري "
            "رقم 14 لسنة 2025، وليس استشارة قانونية. للحصول على استشارة قانونية "
            "بخصوص حالتك، يرجى الرجوع إلى محامٍ مرخّص."
        ),
        "placeholder": "اكتب سؤالك عن قانون العمل...",
        "spinner": "جارٍ البحث في القانون...",
        "loading": "جارٍ تحميل النظام لأول مرة، قد يستغرق ذلك دقيقة أو أكثر...",
        "about_title": "عن المساعد",
        "about_text": (
            "يبحث في نصوص قانون العمل رقم 14 لسنة 2025 ويجيب بالاستناد إلى المواد "
            "المسترجعة فقط، مع ذكر أرقام المواد."
        ),
        "new_chat": "🗑️ محادثة جديدة",
        "more_topics": "موضوعات أخرى للاستكشاف",
        "start": "ابدأ بسؤال",
        "source": "المصدر:",
        "article": "مادة {n}",
        "err_conn": (
            "تعذر الاتصال بالخادم. تأكد من تشغيل الخادم "
            "(uvicorn src.api.main:app --reload) ثم أعد المحاولة."
        ),
        "err_timeout": "استغرق الطلب وقتًا طويلاً، يرجى المحاولة مرة أخرى.",
        "warn_429": "عدد الطلبات كبير حاليًا، يرجى المحاولة بعد قليل",
        "err_generic": "حدث خطأ غير متوقع.",
        "internal_warn": (
            "تنبيه داخلي: تم الاستشهاد بمواد لم يتم استرجاعها فعليًا: {x}"
        ),
        "examples": [
            "ما هي مدة الإجازة السنوية للعامل؟",
            "ما هي مدة فترة الاختبار؟",
            "متى يحق للعامل الحصول على إجازة وضع؟",
            "ما هي شروط إنهاء عقد العمل؟",
        ],
        "side": [
            "ما هي ساعات العمل اليومية القصوى؟",
            "كيف يتم احتساب أجر العمل الإضافي؟",
            "ما هي حقوق العاملة بعد الولادة؟",
            "متى يحق لصاحب العمل فصل العامل؟",
            "ما هو الحد الأدنى للأجور؟",
        ],
        "head_font": "'Amiri', serif",
        "body_font": "'Tajawal', sans-serif",
        "h1": "2.3rem",
        "answer": "1.2rem",
    },

    "en": {
        "dir": "ltr",
        "page_title": "Egyptian Labor Law Information Assistant",
        "title": "Egyptian Labor Law Information Assistant",
        "subtitle": "Labor Law No. 14 of 2025",
        "disclaimer": (
            "This system provides general legal information based on Egyptian Labor "
            "Law No. 14 of 2025. It is not legal advice. For advice about your "
            "specific situation, please consult a licensed lawyer."
        ),
        "placeholder": "Ask a question about the labor law...",
        "spinner": "Searching the law...",
        "loading": "Loading the system for the first time - this can take a minute or more...",
        "about_title": "About",
        "about_text": (
            "Searches the text of Labor Law No. 14 of 2025 and answers only from "
            "the retrieved articles, citing article numbers."
        ),
        "new_chat": "🗑️ New conversation",
        "more_topics": "More topics to explore",
        "start": "Start with a question",
        "source": "Source:",
        "article": "Article {n}",
        "err_conn": (
            "Could not reach the server. Make sure it is running "
            "(uvicorn src.api.main:app --reload) and try again."
        ),
        "err_timeout": "The request took too long. Please try again.",
        "warn_429": "Too many requests right now. Please try again shortly.",
        "err_generic": "An unexpected error occurred.",
        "abstain": (
            "This is not covered in the indexed text of Labor Law No. 14 of 2025, "
            "so I can't answer it. Please rephrase your question or consult a licensed lawyer."
        ),
        "internal_warn": (
            "Internal warning: cited articles that were not actually retrieved: {x}"
        ),
        "examples": [
            "What is the annual leave duration for an employee?",
            "How long is the probation period?",
            "When is a worker entitled to maternity leave?",
            "What are the conditions for terminating an employment contract?",
        ],
        "side": [
            "What are the maximum daily working hours?",
            "How is overtime pay calculated?",
            "What are a mother's rights after childbirth?",
            "When can an employer dismiss a worker?",
            "What is the minimum wage?",
        ],
        "head_font": "'Source Serif 4', Georgia, serif",
        "body_font": "'Inter', sans-serif",
        "h1": "2.1rem",
        "answer": "1.1rem",
    },
}


# ---------------------------------------------------------------- session state

if "lang" not in st.session_state:
    st.session_state.lang = "ar"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending" not in st.session_state:
    st.session_state.pending = None


t = TEXT[st.session_state.lang]


# ---------------------------------------------------------------- page config

st.set_page_config(
    page_title=t["page_title"],
    page_icon="⚖️",
    layout="centered",
)


# ---------------------------------------------------------------- styling

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Tajawal:wght@400;500;700&family=Source+Serif+4:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');


/* -------------------------------------------------------------
   Theme variables
   ------------------------------------------------------------- */

:root {
    --brass: #C49A3C;
    --lapis: #4E8FCB;
    --line: color-mix(in srgb, currentColor 22%, transparent);
    --tint: color-mix(in srgb, currentColor 6%, transparent);
}


/* -------------------------------------------------------------
   Direction
   ------------------------------------------------------------- */

[data-testid="stMain"],
section.main,
[data-testid="stSidebarContent"],
[data-testid="stBottom"] {
    direction: __DIR__;
    text-align: start;
}


/* -------------------------------------------------------------
   Fonts
   ------------------------------------------------------------- */

html,
body,
.stApp,
p,
li,
label,
button,
input,
textarea,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {
    font-family: __BODY__;
}

h1,
h2,
h3,
h4 {
    font-family: __HEAD__;
}


/* -------------------------------------------------------------
   Hide Streamlit chrome
   ------------------------------------------------------------- */

#MainMenu,
footer,
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* -------------------------------------------------------------
   Sidebar controls
   ------------------------------------------------------------- */

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: inherit !important;
}


/* -------------------------------------------------------------
   Main layout
   ------------------------------------------------------------- */

.block-container {
    padding-top: 2rem;
    max-width: 780px;
}


/* -------------------------------------------------------------
   Masthead
   ------------------------------------------------------------- */

.masthead {
    border-bottom: 3px double var(--brass);
    padding-bottom: .9rem;
    margin-bottom: 1rem;
}

.masthead h1 {
    font-family: __HEAD__;
    font-size: __H1__;
    margin: 0;
    line-height: 1.3;
}

.masthead p {
    margin: .2rem 0 0;
    color: var(--lapis);
    font-size: 1rem;
}


/* -------------------------------------------------------------
   Disclaimer
   ------------------------------------------------------------- */

.notice {
    font-size: .85rem;
    opacity: .85;
    background: var(--tint);
    border: 1px solid var(--line);
    border-inline-start: 4px solid var(--lapis);
    border-radius: 6px;
    padding: .6rem .9rem;
    margin-bottom: 1.2rem;
}


/* -------------------------------------------------------------
   Assistant message
   ------------------------------------------------------------- */

/*
   The answer is rendered directly inside Streamlit's native
   assistant chat message.

   This means the text, tables, bullets, and headings all stay
   inside the assistant bubble.
*/

[data-testid="stChatMessage"] {
    direction: __DIR__;
}


/* -------------------------------------------------------------
   Assistant typography
   ------------------------------------------------------------- */

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    font-family: __HEAD__;
    font-size: __ANSWER__;
    line-height: 1.9;
}


/* -------------------------------------------------------------
   Assistant paragraphs
   ------------------------------------------------------------- */

[data-testid="stChatMessage"] p {
    margin-top: 0;
    margin-bottom: .8rem;
}


/* -------------------------------------------------------------
   Assistant headings
   ------------------------------------------------------------- */

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    font-family: __HEAD__;
    margin-top: 1rem;
    margin-bottom: .6rem;
}


/* -------------------------------------------------------------
   Assistant lists
   ------------------------------------------------------------- */

[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    margin-top: .4rem;
    margin-bottom: .9rem;
    padding-inline-start: 1.5rem;
}

[data-testid="stChatMessage"] li {
    margin-bottom: .35rem;
}


/* -------------------------------------------------------------
   Assistant tables
   ------------------------------------------------------------- */

[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: .92rem;
    display: block;
    overflow-x: auto;
}

[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    border: 1px solid var(--line);
    padding: .65rem .7rem;
    vertical-align: top;
    text-align: start;
}

[data-testid="stChatMessage"] th {
    font-weight: 700;
    background: var(--tint);
}

[data-testid="stChatMessage"] tr:nth-child(even) {
    background: color-mix(in srgb, currentColor 3%, transparent);
}


/* -------------------------------------------------------------
   Assistant bold text
   ------------------------------------------------------------- */

[data-testid="stChatMessage"] strong {
    font-weight: 700;
}


/* -------------------------------------------------------------
   Sources
   ------------------------------------------------------------- */

.sources {
    margin-top: .9rem;
    padding-top: .7rem;
    border-top: 1px dashed var(--line);
    font-size: .9rem;
}

.chip {
    display: inline-block;
    margin: 0 .2rem .3rem .2rem;
    padding: .05rem .7rem;
    background: color-mix(in srgb, var(--lapis) 20%, transparent);
    border: 1px solid color-mix(in srgb, var(--lapis) 55%, transparent);
    border-radius: 999px;
    font-weight: 500;
}


/* -------------------------------------------------------------
   User message
   ------------------------------------------------------------- */

.user-q {
    font-family: __BODY__;
    font-size: 1.05rem;
    font-weight: 500;
}


/* -------------------------------------------------------------
   Chat input
   ------------------------------------------------------------- */

[data-testid="stChatInput"] textarea {
    direction: __DIR__;
    text-align: start;
}


/* -------------------------------------------------------------
   Buttons
   ------------------------------------------------------------- */

.stButton > button {
    border-radius: 6px;
    border: 1px solid var(--line);
    background: var(--tint);
    color: inherit;
    text-align: start;
    width: 100%;
    padding: .6rem .9rem;
}

.stButton > button:hover {
    border-color: var(--lapis);
    color: var(--lapis);
}


/* -------------------------------------------------------------
   Language button
   ------------------------------------------------------------- */

.language-button button {
    width: auto !important;
    min-width: 110px;
}


/* -------------------------------------------------------------
   Responsive tables
   ------------------------------------------------------------- */

@media (max-width: 700px) {

    [data-testid="stChatMessage"] table {
        font-size: .85rem;
    }

    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] td {
        padding: .5rem;
    }

}

</style>
"""


st.markdown(
    CSS.replace("__DIR__", t["dir"])
    .replace("__BODY__", t["body_font"])
    .replace("__HEAD__", t["head_font"])
    .replace("__H1__", t["h1"])
    .replace("__ANSWER__", t["answer"]),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- API helper


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """
    Direct mode only: load the embedding model, Chroma collection and Groq
    client ONCE per server process and share them across all sessions.
    """

    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"),
    )

    # Streamlit Community Cloud keeps secrets in st.secrets; make the Groq
    # key visible to the Groq client (which reads the environment).
    if not os.getenv("GROQ_API_KEY"):
        try:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    from ask import load_resources

    resources = load_resources()

    # Log peak memory so the host's RAM limit can be checked against reality.
    try:
        import resource  # Linux/macOS only

        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[memory] peak RSS after loading pipeline: {peak_mb:.0f} MB", flush=True)
    except Exception:
        pass

    return resources


@st.cache_data(max_entries=500, show_spinner=False)
def _answer_direct(question: str, lang: str = "ar") -> dict:
    """
    Direct mode: run the Phase 2 pipeline in-process. Successful answers are
    cached (repeat questions are instant); exceptions are never cached.
    """

    from ask import ask

    embed_model, collection, groq_client = _load_pipeline()
    return ask(embed_model, collection, groq_client, question, verbose=False, lang=lang)


def ask_api(question: str, lang: str = "ar") -> dict:
    """
    Answer a question, via the FastAPI backend ("api" mode, default) or
    in-process ("direct" mode).

    Either way the backend logic remains responsible for:
    - retrieval
    - generation
    - citations
    - abstention
    - answer structure
    """

    if RAG_MODE == "direct":

        try:
            result = _answer_direct(" ".join(question.split()), lang)

        except Exception as e:

            if getattr(e, "status_code", None) == 429:
                return {
                    "kind": "warning",
                    "key": "warn_429",
                }

            print(f"[ERROR] direct ask failed: {type(e).__name__}: {e}", flush=True)

            return {
                "kind": "error",
                "key": "err_generic",
            }

        return {
            "result": result,
        }

    try:

        response = requests.post(
            API_URL,
            json={"question": question, "lang": lang},
            timeout=30,
        )

    except requests.exceptions.ConnectionError:

        return {
            "kind": "error",
            "key": "err_conn",
        }

    except requests.exceptions.Timeout:

        return {
            "kind": "error",
            "key": "err_timeout",
        }


    detail = None

    try:
        detail = response.json().get("detail")
    except ValueError:
        pass


    if response.status_code == 429:

        return {
            "kind": "warning",
            "key": "warn_429",
            "detail": detail,
        }


    if response.status_code != 200:

        return {
            "kind": "error",
            "key": "err_generic",
            "detail": detail,
        }


    return {
        "result": response.json()
    }


# ---------------------------------------------------------------- interaction helpers


def set_pending(q: str) -> None:
    """
    Store an example/sidebar question so it can be submitted
    on the next Streamlit rerun.
    """

    st.session_state.pending = q


def clear_chat() -> None:
    """Clear the current conversation."""

    st.session_state.messages = []


def toggle_lang() -> None:
    """
    Toggle between Arabic and English.

    Arabic -> English
    English -> Arabic
    """

    if st.session_state.lang == "ar":
        st.session_state.lang = "en"
    else:
        st.session_state.lang = "ar"


# ---------------------------------------------------------------- assistant renderer


def render_assistant(msg: dict) -> None:
    """
    Render an assistant response.

    The backend answer is rendered directly as Markdown.

    This is important because Streamlit needs the original Markdown
    in order to correctly display:

    - tables
    - bullet points
    - numbered lists
    - bold text
    - headings
    - paragraphs
    """

    # -------------------------------------------------------------
    # Errors / warnings
    # -------------------------------------------------------------

    if "kind" in msg:

        text = msg.get("detail") or t[msg["key"]]

        if msg["kind"] == "error":
            st.error(text)
        else:
            st.warning(text)

        return


    result = msg["result"]


    # -------------------------------------------------------------
    # Abstention
    # -------------------------------------------------------------

    if result["is_abstention"]:

        st.warning(
            t["abstain"] if st.session_state.lang == "en" else result["answer"]
        )

        return


    # -------------------------------------------------------------
    # Main answer
    # -------------------------------------------------------------

    # Do not escape or convert the answer to HTML.
    #
    # The Markdown is rendered directly inside the native
    # Streamlit assistant message bubble.

    st.markdown(
        result["answer"].strip(),
        unsafe_allow_html=False,
    )


    # -------------------------------------------------------------
    # Sources
    # -------------------------------------------------------------

    if result["cited_articles"]:

        chips = "".join(
            f'<span class="chip">'
            f'{html.escape(t["article"].format(n=n))}'
            f'</span>'
            for n in result["cited_articles"]
        )

        st.markdown(
            f'<div class="sources">'
            f'{t["source"]} {chips}'
            f'</div>',
            unsafe_allow_html=True,
        )


    # -------------------------------------------------------------
    # Internal citation warning
    # -------------------------------------------------------------

    if result["hallucinated_citations"]:

        st.error(
            t["internal_warn"].format(
                x=result["hallucinated_citations"]
            )
        )


# ---------------------------------------------------------------- language button


# ONE language button only.
#
# Arabic page:
#       [ English ]
#
# English page:
#       [ العربية ]

other_lang_label = (
    "English"
    if st.session_state.lang == "ar"
    else "العربية"
)


lang_col, _ = st.columns([1, 5])


with lang_col:

    st.button(
        other_lang_label,
        key="language_switch",
        on_click=toggle_lang,
        type="secondary",
    )


# ---------------------------------------------------------------- masthead


st.markdown(
    f'<div class="masthead">'
    f'<h1>⚖️ {t["title"]}</h1>'
    f'<p>{t["subtitle"]}</p>'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- disclaimer


st.markdown(
    f'<div class="notice">{t["disclaimer"]}</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- warm-up (direct mode)

# In direct mode the first visitor triggers the model load. Do it here, with
# a visible message, instead of leaving the first question hanging. After the
# first load it is cached for every later visitor and question.

if RAG_MODE == "direct":

    with st.spinner(t["loading"]):

        _load_pipeline()


# ---------------------------------------------------------------- question input


typed = st.chat_input(
    t["placeholder"]
)


question = (
    typed
    or st.session_state.pending
    or ""
).strip()


st.session_state.pending = None


if question:

    # -------------------------------------------------------------
    # Store user question
    # -------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # -------------------------------------------------------------
    # Ask backend
    # -------------------------------------------------------------

    with st.spinner(t["spinner"]):

        reply = ask_api(question, st.session_state.lang)


    # -------------------------------------------------------------
    # Store assistant response
    # -------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            **reply,
        }
    )


# ---------------------------------------------------------------- sidebar


with st.sidebar:

    st.markdown(
        f"### {t['about_title']}"
    )

    st.write(
        t["about_text"]
    )


    # Show new conversation button only when a conversation exists.

    if st.session_state.messages:

        st.button(
            t["new_chat"],
            on_click=clear_chat,
            key="new_chat",
        )


    st.markdown(
        f"### {t['more_topics']}"
    )


    # Sidebar example questions.

    for i, q in enumerate(t["side"]):

        st.button(
            q,
            key=f"side_{i}",
            on_click=set_pending,
            args=(q,),
        )


# ---------------------------------------------------------------- example questions


if not st.session_state.messages:

    st.markdown(
        f"#### {t['start']}"
    )

    cols = st.columns(2)


    for i, q in enumerate(t["examples"]):

        cols[i % 2].button(
            q,
            key=f"main_{i}",
            on_click=set_pending,
            args=(q,),
        )


# ---------------------------------------------------------------- conversation history


for msg in st.session_state.messages:

    # -------------------------------------------------------------
    # User message
    # -------------------------------------------------------------

    if msg["role"] == "user":

        with st.chat_message("user"):

            st.markdown(
                f'<div class="user-q">'
                f'{html.escape(msg["content"])}'
                f'</div>',
                unsafe_allow_html=True,
            )


    # -------------------------------------------------------------
    # Assistant message
    # -------------------------------------------------------------

    else:

        with st.chat_message(
            "assistant",
            avatar="⚖️",
        ):

            render_assistant(msg)
