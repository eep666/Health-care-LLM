import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Healthcare Chat",
    page_icon="🤖",
    layout="wide"
)

# --- Securely Get API Key ---
# This function gets the key from Streamlit secrets when deployed,
# or from a .env file locally. It's not displayed to the user.
def get_api_key():
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()

# --- Sidebar for Generation Parameters ---
with st.sidebar:
    st.title("Configuration")
    st.caption("Tweak the generation parameters for the model.")

    if not api_key:
        st.error("GEMINI_API_KEY is not set. Please add it to your secrets.")
        st.stop()

    model_name = st.selectbox("Model", ["gemini-1.5-flash"])
    system_instruction = st.text_area(
        "System Instruction",
        value="You are a concise, helpful assistant that helps the user get answers on Healthcare related topics only.",
        height=100
    )

    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 100, 40, 1)
    max_tokens = st.slider("Max output tokens", 32, 2048, 512, 32)


# --- Helper Function for Model ---
@st.cache_resource(show_spinner="Connecting to the model...")
def get_model(_api_key, model_name, system_instruction):
    """Create and cache a GenerativeAI Model handle."""
    try:
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
        return model
    except Exception as e:
        st.error(f"Failed to configure or create the model: {e}")
        st.stop()

# --- Chat Logic ---
st.title("🤖 Healthcare Chat")
st.caption("Ask me anything about general healthcare topics.")

# Initialize chat
model = get_model(api_key, model_name, system_instruction)

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_prompt = st.chat_input("Ask something about healthcare...", key="chat_input")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    gen_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_tokens,
    }

    # Generate and stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            responses = st.session_state.chat.send_message(
                user_prompt,
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
