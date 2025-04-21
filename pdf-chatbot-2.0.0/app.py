import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pdfplumber
from pathlib import Path
from docx import Document

# Set up the UI
st.set_page_config(page_title="Document Chatbot", page_icon="📄")
st.title("📄 Document Chatbot")
st.markdown("Upload your document and chat with its contents")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# File uploader with validation


def validate_file(file):
    allowed_types = ["application/pdf", "text/plain",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    return file.type in allowed_types


uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "txt", "docx"],
    help="Only PDF, TXT, and DOCX files are allowed"
)

# Function to extract text from PDF/TXT/DOCX


def extract_text(file_path):
    """
    Extracts text from PDF/TXT/DOCX files.
    Returns: List of text chunks (pages/sections) or None if failed
    """
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # PDF handling
        if file_path.lower().endswith('.pdf'):
            text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text(x_tolerance=3)  # PPT-friendly
                    if content:
                        text.append(content)
            return text if text else None

        # TXT handling
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return [f.read()]

        # DOCX handling
        elif file_path.lower().endswith('.docx'):
            doc = Document(file_path)
            return ["\n".join(para.text for para in doc.paragraphs)]

        else:
            raise ValueError("Unsupported file type. Use PDF/TXT/DOCX")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


if uploaded_file and validate_file(uploaded_file):
    # Save file and show success
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Format: YYYYMMDDHHMMSS
    save_path = os.path.join(
        "documents", f"{timestamp}_{uploaded_file.name}")
    # creates folder if it doesn't exists
    os.makedirs("documents", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.uploaded_file = uploaded_file
    if uploaded_file.size == 0:
        st.error("❌ The uploaded file is empty. Please upload a valid file.")
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Execute extraction
    result = extract_text(save_path)
    if result is None:
        st.error("❌ Failed to extract text from the document.")
    else:
        st.success("✅ Text extracted successfully!")
    st.session_state.messages = []  # Clear previous chat if new file is uploaded
elif uploaded_file and not validate_file(uploaded_file):
    st.error("❌ Invalid file type. Please upload PDF, TXT, or DOCX only.")

# Chat interface
if st.session_state.uploaded_file:
    st.divider()
    st.subheader(f"Chat with {st.session_state.uploaded_file.name}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response (placeholder - replace with your model)
        ai_response = f"I received your question about {st.session_state.uploaded_file.name}. (This is a placeholder response)"

        # Add AI response to chat
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response})
else:
    st.info("Please upload a document to begin chatting")
