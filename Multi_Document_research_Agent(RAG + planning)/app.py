import streamlit as st
from mcp import send_mcp_message, receive_mcp_message
from ingestion_agent import process_document
from retrieval_agent import store_embeddings, retrieve_chunks
from llm_response_agent import generate_response
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename="app.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Coordinator function to manage uploads
def coordinate_upload(file, filename):
    success = process_document(file, filename)
    if success:
        message = receive_mcp_message("RetrievalAgent")
        if message:
            store_embeddings(message["payload"]["chunks"], message["payload"]["document_name"])
        return success
    return False

# Coordinator function to manage queries
def coordinate_query(query):
    top_chunks = retrieve_chunks(query)
    message = receive_mcp_message("LLMResponseAgent")
    if message:
        response, sources = generate_response(message["payload"]["query"], message["payload"]["top_chunks"])
        return response, sources
    return None, None

# Helper: Format response as bullet points with sources
def format_response(response_text, sources):
    bullets = response_text.strip().split("\n")
    formatted = ""
    for b in bullets:
        if b.strip():
            formatted += f"- {b.strip()}\n"
    if sources:
        formatted += "\n**Sources:** " + ", ".join(sources)
    return formatted

# Streamlit UI
def main():
    st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
    st.title("Agentic RAG Chatbot")
    st.write("Upload multiple documents and ask questions about their content. Responses include sources.")

    # Document upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "pptx", "csv", "docx", "txt", "md"]
    )
    if uploaded_files:
        for file in uploaded_files:
            if coordinate_upload(file, file.name):
                st.success(f"Successfully processed {file.name}")
            else:
                st.error(f"Failed to process {file.name}")

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(format_response(message["content"], message.get("sources", [])))
            else:
                st.markdown(message["content"])

    # User prompt
    if prompt := st.chat_input("Ask a question about the documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response, sources = coordinate_query(prompt)
        if response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            with st.chat_message("assistant"):
                st.markdown(format_response(response, sources))
        else:
            st.error("No relevant information found.")

if __name__ == "__main__":
    main()
