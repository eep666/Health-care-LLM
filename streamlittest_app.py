import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from a .env file for local development
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Healthcare LLM",
    page_icon="🤖",
    layout="wide"
)

# --- Securely Get API Key ---
def get_api_key():
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()

# --- Helper Function for Placeholder Image ---
@st.cache_resource(show_spinner=False)
def get_placeholder_image():
    """Create and cache a simple placeholder image."""
    return Image.new("RGB", (200, 100), color=(240, 240, 240))

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Configuration")
    st.caption("Tweak the model and its parameters.")

    if not api_key:
        st.error("GEMINI_API_KEY is not set. Please add it to your secrets.")
        st.stop()

    # Model selection with both Pro and Flash options
    model_name = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash"])

    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    
    st.write("---")
    st.subheader("Image Uploader")
    uploaded_file = st.file_uploader("Upload an image to ask questions about it.", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = None
    
    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Helper Function for Model ---
@st.cache_resource(show_spinner="Connecting to the model...")
def get_model(_api_key, model_name):
    """Create and cache a GenerativeAI Model handle."""
    try:
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to configure or create the model: {e}")
        st.stop()

# --- Main Chat Logic ---
st.title("🤖 Healthcare LLM")
st.caption("Your intelligent assistant for healthcare questions.")

# Initialize chat
model = get_model(api_key, model_name)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display a welcome message and example prompts if the chat is new
if not st.session_state.messages:
    st.info("Welcome! I'm here to help with your healthcare questions. Try asking one of the examples below or type your own question.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What are the symptoms of diabetes?"):
            st.session_state.messages.append({"role": "user", "content": "What are the symptoms of diabetes?"})
            st.rerun()
    with col2:
        if st.button("Explain the difference between a virus and bacteria"):
            st.session_state.messages.append({"role": "user", "content": "Explain the difference between a virus and bacteria"})
            st.rerun()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_prompt = st.chat_input("Ask a question about health...", key="chat_input")
if user_prompt:
    # Prepare the content to send to the model (text and optional image)
    content_to_send = [user_prompt]
    if image:
        content_to_send.append(image)

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    gen_config = {
        "temperature": temperature,
        "top_p": top_p,
    }

    # Generate and stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # The model call is now more robust, handling both text and images
            responses = model.generate_content(
                content_to_send,
                generation_config=gen_config,
                stream=True
            )
            for response in responses:
                full_response += response.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I encountered an error."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
