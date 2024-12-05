import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utlis import save_chat_history_json, get_timestamp, load_chat_history_json
from image_handler import handle_image
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from html_templates import get_bot_template, get_user_template, css
import yaml
import os

# Load configuration settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dynamically load the appropriate LLM chain based on the current session state
def load_chain(chat_history):
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain(chat_history) # Chain for handling PDF-related chats
    return load_normal_chain(chat_history)  # Chain for handling normal chats

# Clear the input field if the user has not entered any text
# Clear input field after processing user input
def clear_input_field():
    if st.session_state.user_question == "":
        st.session_state.user_question = st.session_state.user_input
        st.session_state.user_input = ""

# Set the state to indicate input should be sent and clear the input field
def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

# Toggle PDF chat mode on
def toggle_pdf_chat():
    st.session_state.pdf_chat = True

# Save the chat history to a JSON file
def save_chat_history():
    if st.session_state.history != []:
        # Generate a unique session key based on the timestamp
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            # Save to the existing session file
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)

# Main function defining the Streamlit app
def main():
    # App title and UI setup
    st.title("Multimodal Local Chat App")
    st.write(css, unsafe_allow_html=True) # Apply custom CSS
    
    # Sidebar for session management and file uploads
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    # Initialize session state variables
    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # Display session selection dropdown in the sidebar
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox(
        "Select a chat session", chat_sessions, key="session_key", index=index
        )
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)

    # Load chat history for the selected session
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(
            config["chat_history_path"] + st.session_state.session_key
            )
    else:
        st.session_state.history = []

    # Initialize chat history using LangChain's Streamlit integration
    chat_history = StreamlitChatMessageHistory(key="history")
    
    # Main input field for user text input
    user_input = st.text_input(
        "Type your message here", key="user_input", on_change=set_send_input
    )

    # UI for voice recording and send button
    voice_recording_column, send_button_column = st.columns(2)
    chat_container = st.container()
    with voice_recording_column:
        voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    # Sidebar options for uploading files
    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a pdf file", accept_multiple_files=True, key="pdf_upload", type=["pdf"], on_change=toggle_pdf_chat
    )

    # Handle PDF uploads and add them to the database
    if uploaded_pdf:
        with st.spinner("Processing pdf..."):
            add_documents_to_db(uploaded_pdf)

    # Process uploaded or recorded audio for transcription
    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain(chat_history)
        llm_chain.run("Summarize this text: " + transcribed_audio)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain(chat_history)
        llm_chain.run(transcribed_audio)

    # Handle user text input and image processing
    if send_button or st.session_state.send_input:
        if uploaded_image:
            with st.spinner("Processing image..."):
                user_message = "Describe this image in detail please."
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_answer = handle_image(uploaded_image.getvalue(), user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)



        if st.session_state.user_question != "":
            llm_chain = load_chain(chat_history)
            llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""

        st.session_state.send_input = False

    # Display chat history in reverse chronological order
    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in reversed(chat_history.messages):
                if message.type == "human":
                    st.write(get_user_template(message.content), unsafe_allow_html=True)
                else:
                    st.write(get_bot_template(message.content), unsafe_allow_html=True)

    # Save the updated chat history
    save_chat_history()

# Run the app
if __name__ == "__main__":
    main()