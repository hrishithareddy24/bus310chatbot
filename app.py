import os
import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Optional: HuggingFace (free)
from huggingface_hub import InferenceClient

# Optional: OpenAI (paid) via LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === Load environment ===
load_dotenv()

st.set_page_config(page_title="BUS 310 Tutor", layout="wide")
st.title("📘 BUS 310 Tutor ")
st.markdown("""
**George Mason University – BUS 310**  
**Prof. Fatou Diouf · TA: Hrishitha Reddy Likki**  
---
This tutor answers **only** from uploaded syllabus, lecture slides, rubrics, and assignments.  
Put PDFs in `./data` → click **Rebuild Index**.
""")

DATA_PATH = Path("data")
VECTOR_PATH = Path("vectorstore")

# =========================
# Settings / Toggles
# =========================
# Set in .env:
# USE_OPENAI=1  (paid, fast)
# USE_OPENAI=0  (free HF)
USE_OPENAI = os.getenv("USE_OPENAI", "0").strip() == "1"

# For paid OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

# Guardrail thresholds (tweak in .env if needed)
# If the retrieved context is too small, we refuse.
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "250"))
# FAISS distance threshold: smaller = more similar. If too strict, bot may refuse more often.
DIST_THRESHOLD = float(os.getenv("DIST_THRESHOLD", "1.8"))

# =========================
# Helper functions
# =========================
def is_definition_question(q: str) -> bool:
    starters = ["what is", "define", "what does", "meaning of"]
    q = (q or "").strip().lower()
    return any(q.startswith(s) for s in starters)

def clip_to_n_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:n]).strip()

def build_system_prompt(user_question: str):
    if is_definition_question(user_question):
        return (
            "You are a BUS 310 tutor at George Mason University. "
            "The student is asking a DEFINITION question. "
            "Answer in EXACTLY 1–2 concise sentences. "
            "Do NOT include examples unless explicitly asked. "
            "Use ONLY the context provided below. "
            "Extract the definition and REWRITE it in your own words (do not paste long context). "
            "If the definition is not found, reply exactly: "
            "'I couldn’t find this definition in the BUS 310 materials.'"
        )
    else:
        return (
            "You are a BUS 310 tutor at George Mason University. "
            "Answer the student's question using ONLY the context provided below. "
            "Respond in a maximum of 4–5 sentences (one paragraph). "
            "Be clear, focused, and exam-oriented. "
            "Do NOT generate a multi-turn dialogue. "
            "If the answer is not found, reply exactly: "
            "'I couldn’t find this in the BUS 310 materials.'"
        )

# =========================
# Load PDFs
# =========================
def load_all_pdfs():
    pdfs = list(DATA_PATH.glob("*.pdf"))
    if not pdfs:
        st.warning("⚠️ No PDFs found in ./data")
        return []
    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

# =========================
# Build Vectorstore (cached)
# =========================
@st.cache_resource
def _build_vectorstore(_docs, batch_size=64):
    if not _docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(_docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pbar = st.progress(0.0)
    vs = None
    total = len(chunks)

    for i in range(0, total, batch_size):
        part = chunks[i:i+batch_size]
        if vs is None:
            vs = FAISS.from_documents(part, emb)
        else:
            vs.add_documents(part)

        progress_value = min(1.0, (i + len(part)) / total)
        pbar.progress(progress_value)

    VECTOR_PATH.mkdir(exist_ok=True)
    vs.save_local(str(VECTOR_PATH))
    pbar.progress(1.0)
    st.success("✅ Index built successfully!")
    return vs

@st.cache_resource
def load_vectorstore():
    if (VECTOR_PATH / "index.faiss").exists():
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(str(VECTOR_PATH), emb, allow_dangerous_deserialization=True)
    return None

# =========================
# Sidebar
# =========================
st.sidebar.header("Course Files")
st.sidebar.write("📁 Put PDFs in the `./data` folder.")
st.sidebar.markdown("---")
st.sidebar.write("**Model mode:** " + ("✅ Paid OpenAI" if USE_OPENAI else "🆓 Free HuggingFace"))

if st.sidebar.button("🔁 Rebuild Index"):
    st.cache_resource.clear()
    with st.spinner("Building index from PDFs…"):
        docs = load_all_pdfs()
        st.session_state.vs = _build_vectorstore(docs)

# =========================
# Ensure vectorstore
# =========================
if "vs" not in st.session_state:
    st.session_state.vs = load_vectorstore()
    if st.session_state.vs:
        st.success("✅ Vectorstore loaded successfully!")
    else:
        st.warning("⚠️ No index found. Click ‘Rebuild Index’ to create one.")

# =========================
# Chat
# =========================
prompt = st.chat_input("Ask a BUS 310 question (e.g. grading policy, lesson topics)…")

if prompt:
    if st.session_state.vs is None:
        st.warning("⚠️ Please upload PDFs and rebuild index first.")
    else:
        st.chat_message("user").write(prompt)

        with st.spinner("🔍 Searching BUS 310 materials…"):
            # Use similarity_search_with_score so we can reject weak matches
            results = st.session_state.vs.similarity_search_with_score(prompt, k=6)

            # Keep only good matches
            good = [(d, s) for (d, s) in results if s <= DIST_THRESHOLD]
            top_docs = [d for (d, _) in good][:4]
            context = "\n\n".join(d.page_content for d in top_docs).strip()

        # Hard refusal if retrieval is weak
        if len(context) < MIN_CONTEXT_CHARS:
            st.chat_message("assistant").write("I couldn’t find this in the BUS 310 materials.")
        else:
            system_prompt = build_system_prompt(prompt)

            # ----- Paid OpenAI path (fast) -----
            if USE_OPENAI:
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    st.error("🚨 Missing OPENAI_API_KEY in your .env file.")
                else:
                    try:
                        llm = ChatOpenAI(
                            model=OPENAI_MODEL,
                            temperature=0.2,
                            api_key=openai_key,
                        )
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {prompt}"),
                        ]
                        answer = llm.invoke(messages).content

                        # Hard cap length
                        answer = clip_to_n_sentences(answer, 2 if is_definition_question(prompt) else 5)

                        st.chat_message("assistant").write(answer)
                    except Exception as e:
                        st.error(f"Model error: {e}")

            # ----- Free HuggingFace path (slower) -----
            else:
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not hf_token:
                    st.error("🚨 Missing Hugging Face token! Add HUGGINGFACEHUB_API_TOKEN=... to your .env file.")
                else:
                    try:
                        client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=hf_token)

                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"},
                        ]

                        response = client.chat_completion(
                            model="HuggingFaceH4/zephyr-7b-beta",
                            messages=messages,
                            temperature=0.3,
                        )
                        answer = response.choices[0].message["content"]

                        # Hard cap length
                        answer = clip_to_n_sentences(answer, 2 if is_definition_question(prompt) else 5)

                        st.chat_message("assistant").write(answer)
                    except Exception as e:
                        st.error(f"Model error: {e}")

st.markdown("---")
st.caption("© 2026 Hrishitha Reddy Likki · George Mason University · BUS 310 Tutor")
