import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.conversation.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from openai import OpenAI

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")
DATA_DIR = "__data__"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Use your API key securely (environment variable recommended)
OPENROUTER_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)

# -----------------------------
# SESSION STATE SETUP
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rubric_text" not in st.session_state:
    st.session_state.rubric_text = ""


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_pdf_text(pdf_docs):
    """Extract text from one or more PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    """Split text into manageable chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def get_vector_stores(text_chunks):
    """Convert text chunks into vector embeddings and store in FAISS."""
    embeddings = OpenAIEmbeddings(
        model="mistralai/mistral-embed",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_KEY
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    """Create conversational retrieval chain for chat."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def extract_text(pdf_file):
    """Extract raw text from a single PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def _essay_grade(essay):
    """Grade an essay based on rubric text in English."""
    messages = [
        {
            "role": "system",
            "content": "You are an English academic writing evaluator. "
                       "Carefully grade the essay based on the given rubric and respond in English. "
                       "Give a detailed evaluation with section-wise scores and comments following the rubric."
                       + st.session_state.rubric_text,
        },
        {"role": "user", "content": "Essay:\n" + essay},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=1500,
    )

    data = response.choices[0].message.content
    return re.sub(r"\n", "<br>", data)


# ✅ ENHANCED: Chat function that supports PDF + general questions with clean Markdown output
def chat_with_pdf_or_general(question):
    """Answer using PDF context or fallback to general AI (clean formatted response)."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_KEY,
        temperature=0.7,
    )

    try:
        # Try answering based on PDF content if available
        if st.session_state.conversation_chain:
            response = st.session_state.conversation_chain({"question": question})
            answer = response.get("answer", "").strip()

            # If no relevant info found, fall back to general knowledge
            if not answer or "I don't know" in answer:
                general = llm.invoke(
                    f"User asked: '{question}'. "
                    "Answer naturally in English, formatted using Markdown with bullets, bold, and clear sections."
                )
                return general.content.strip()

            return answer

        # No PDF loaded, use general AI directly
        else:
            general = llm.invoke(
                f"User asked: '{question}'. "
                "Respond clearly in English and format neatly using Markdown (use lists, headings, etc.)."
            )
            return general.content.strip()

    except Exception as e:
        return f"⚠️ Error while generating answer: {e}"


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📚 AI Study Assistant")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📄 PDF Chat", "📋 Essay Rubric", "🧠 Essay Grading"],
)


# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("🎓 AI-Powered Study Assistant")
    st.markdown("""
    This app allows you to:
    - Upload PDFs and **chat** with them.
    - Ask **any kind of question**, not just about PDFs.
    - Paste or upload essays for **automatic grading** (based on your rubric).
    - Store your own **grading rubric** for essay evaluation.
    """)


# -----------------------------
# PDF CHAT PAGE (Enhanced)
# -----------------------------
elif page == "📄 PDF Chat":
    st.title("📘 Chat with your PDFs — or ask anything!")

    uploaded_pdfs = st.file_uploader(
        "Upload your PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_pdfs:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(uploaded_pdfs)
            chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_stores(chunks)
            st.session_state.conversation_chain = get_conversation_chain(
                st.session_state.vector_store
            )
        st.success("✅ PDFs processed successfully! You can now chat or ask anything.")

    user_question = st.text_input("💬 Ask something (related to PDF or anything else):")

    if user_question:
        with st.spinner("Thinking..."):
            answer = chat_with_pdf_or_general(user_question)
            st.session_state.chat_history.append(("🧑 User", user_question))
            st.session_state.chat_history.append(("🤖 Bot", answer))

    if st.session_state.chat_history:
        st.subheader("💬 Chat History:")
        for role, msg in st.session_state.chat_history:
            if role == "🧑 User":
                st.markdown(f"**{role}:** {msg}")
            else:
                st.markdown(f"**{role}:**")
                st.markdown(msg, unsafe_allow_html=True)


# -----------------------------
# ESSAY RUBRIC PAGE
# -----------------------------
elif page == "📋 Essay Rubric":
    st.title("📋 Set Essay Grading Rubric")
    st.session_state.rubric_text = st.text_area(
        "Enter your grading rubric below:",
        value=st.session_state.rubric_text,
        height=200,
    )
    if st.button("💾 Save Rubric"):
        st.success("Rubric saved successfully!")


# -----------------------------
# ESSAY GRADING PAGE
# -----------------------------
elif page == "🧠 Essay Grading":
    st.title("🧠 Essay Grading System")

    uploaded_file = st.file_uploader("Upload essay (PDF optional):", type=["pdf"])
    essay_text = st.text_area("Or paste your essay text here:", height=250)

    if st.button("📝 Grade Essay"):
        text_to_grade = ""

        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                text_to_grade = extract_text(uploaded_file)
        elif essay_text.strip():
            text_to_grade = essay_text.strip()

        if not text_to_grade:
            st.warning("⚠️ Please upload or enter essay text before grading.")
        else:
            with st.spinner("Grading essay..."):
                result = _essay_grade(text_to_grade)
                st.markdown("### 🏆 Evaluation Result:")
                st.markdown(result, unsafe_allow_html=True)


