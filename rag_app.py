import os
import base64
import shutil
import time
import streamlit as st
from datetime import datetime
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# ---- App Branding ----
APP_NAME = "⚡ QuickPDF AI"
APP_TAGLINE = "Get answers in seconds."

# ---- Embeddings & LLM ----
embeddings_model = OllamaEmbeddings(model="deepseek-r1:1.5b")

def get_ollama_llm():
    return OllamaLLM(model="deepseek-r1:1.5b")

# ---- Paths ----
DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ---- PDF Loader ----
def load_and_split_documents():
    loader = PyPDFDirectoryLoader(DATA_FOLDER)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ---- Vector Store ----
def create_vector_store(docs):
    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local(INDEX_FOLDER)

def get_vector_store():
    if not os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss")):
        docs = load_and_split_documents()
        if docs:
            create_vector_store(docs)
        else:
            return None
    return FAISS.load_local(INDEX_FOLDER, embeddings_model, allow_dangerous_deserialization=True)

# ---- Voice ----
def speak_text(text, filename="answer.mp3", autoplay=True):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    with open(filename, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode()
    if autoplay:
        st.markdown(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_bytes) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"[Voice recognition failed: {e}]"

# ---- Prompt ----
prompt_template = """
Human: Use the following context to answer the question with a concise and factual response (250 words max).
If the answer is unknown, say "I don't know."

<context>
{context}
</context>

Question: {question}

Assistant:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---- QA ----
def get_answer(query, fun_mode=True):
    db = get_vector_store()
    if db is None:
        return "No documents found. Please upload PDFs first."

    qa = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa({"query": query})
    answer = result["result"]

    if fun_mode:
        speak_text(answer, autoplay=True)

    return answer

# ---- WhatsApp-Style Chat ----
def display_chat(messages, typing=False):
    st.markdown("""
        <style>
        .chat-box {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #f5f5f5;
            scroll-behavior: smooth;
        }
        .message {
            display: flex;
            align-items: flex-end;
            margin-bottom: 12px;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            margin: 0 6px;
        }
        .bubble-user {
            background-color: #DCF8C6;
            color: #000;
            padding: 8px 12px;
            border-radius: 15px;
            max-width: 70%;
            word-wrap: break-word;
            margin-left: auto;
        }
        .bubble-ai {
            background-color: #34B7F1;
            color: #fff;
            padding: 8px 12px;
            border-radius: 15px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .timestamp {
            font-size: 10px;
            color: #ddd;
            margin-top: 3px;
        }
        .typing-dots {
            display: flex;
            gap: 4px;
            align-items: center;
        }
        .typing-dots span {
            display: block;
            width: 6px;
            height: 6px;
            background-color: #fff;
            border-radius: 50%;
            animation: blink 1.4s infinite both;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
            40% { opacity: 1; transform: scale(1); }
        }
        </style>
        <div id="chat-container" class="chat-box">
    """, unsafe_allow_html=True)

    for role, text, timestamp in messages:
        if role == "user":
            st.markdown(f"""
            <div class="message" style="justify-content: flex-end;">
                <div class="bubble-user">{text}<div class="timestamp">{timestamp}</div></div>
                <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/1077/1077012.png">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message">
                <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png">
                <div class="bubble-ai">{text}<div class="timestamp">{timestamp}</div></div>
            </div>
            """, unsafe_allow_html=True)

    if typing:
        st.markdown("""
        <div class="message">
            <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png">
            <div class="bubble-ai">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        </div>
        <script>
            var chatBox = document.getElementById("chat-container");
            chatBox.scrollTop = chatBox.scrollHeight;
        </script>
    """, unsafe_allow_html=True)

# ---- Main App ----
def main():
    st.set_page_config(page_title="QuickPDF AI", page_icon="⚡", layout="centered")
    st.markdown(f"<h1 style='text-align:center;'>{APP_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:gray;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    uploaded_files = st.file_uploader("Upload PDF files:", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            with open(os.path.join(DATA_FOLDER, f.name), "wb") as out_file:
                out_file.write(f.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    # Sidebar controls
    with st.sidebar:
        st.markdown(f"### ⚙️ {APP_NAME} Settings")
        mode = st.radio("Mode", ["Fun Mode (Typing & Sound)", "Fast Mode (Instant Reply)"])
        fun_mode = mode.startswith("Fun")

        if st.button("Create/Update Vector Store"):
            with st.spinner("⏳ Processing documents..."):
                docs = load_and_split_documents()
                if docs:
                    create_vector_store(docs)
                    st.success("Vector store updated!")
                else:
                    st.warning("No PDFs found. Please upload first.")
        if st.button("Delete All Data"):
            shutil.rmtree(DATA_FOLDER, ignore_errors=True)
            shutil.rmtree(INDEX_FOLDER, ignore_errors=True)
            os.makedirs(DATA_FOLDER, exist_ok=True)
            st.session_state.chat = []
            st.success("All data deleted.")

    st.markdown("### 🎤 Ask a Question (Speak or Type)")
    audio_data = mic_recorder(start_prompt="Start Speaking", stop_prompt="Stop", just_once=True)

    recognized_text = ""
    if audio_data:
        recognized_text = transcribe_audio(audio_data["bytes"])
        if recognized_text and not recognized_text.startswith("[Voice recognition failed"):
            st.write(f"**You (Voice):** {recognized_text}")
        else:
            st.error(recognized_text)

    user_query = st.text_input("Or type your question:", value=recognized_text)

    # Handle query (no duplicate messages now)
    if st.button("Send") and user_query:
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.chat.append(("user", user_query, timestamp))

        # Show typing dots only in Fun Mode
        if fun_mode:
            display_chat(st.session_state.chat, typing=True)
            time.sleep(2)

        with st.spinner("⏳ AI is thinking..."):
            answer = get_answer(user_query, fun_mode=fun_mode)

        st.session_state.chat.append(("ai", answer, datetime.now().strftime("%I:%M %p")))

    st.markdown("### 💬 Conversation")
    display_chat(st.session_state.chat)

if __name__ == "__main__":
    main()
