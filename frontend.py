import streamlit as st
import requests
import json

# ==============================================================================
# CONFIGURATION: Set your backend domain and endpoints here
# ==============================================================================
BACKEND_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BACKEND_URL}/v1/chat/completions"
# ==============================================================================

# Setup page layout like LM Studio
st.set_page_config(page_title="LM Studio UI Clone", layout="wide")
st.title("🤖 Local LM Studio Playground")

# Sidebar for Model parameters and configurations
st.sidebar.header("Model Configuration")

# --- Dynamic API Server Health Check ---
backend_alive = False
try:
    # Perform a light check against the base backend route
    response = requests.get(BACKEND_URL, timeout=1.5)
    backend_alive = True
except requests.exceptions.RequestException:
    backend_alive = False

if backend_alive:
    st.sidebar.success("● Connected to Backend")
else:
    st.sidebar.error("○ Offline: Run backend.py")

system_prompt = st.sidebar.text_area(
    "System Prompt", 
    value="You are a helpful, concise AI assistant."
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=2048, value=512)

# Maintain conversational history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Lock down input control if backend isn't responding
chat_placeholder = "Type your message here..." if backend_alive else "Chat disabled: Start backend server in console first."
user_input = st.chat_input(chat_placeholder, disabled=not backend_alive)

# Accept user input
if user_input:
    # Render user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Render assistant placeholder container for real-time streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Send parameters directly to our local FastAPI service
        payload = {
            "message": user_input,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": int(max_tokens)
        }

        try:
            # Open a streaming POST request using the domain variable configuration
            with requests.post(CHAT_ENDPOINT, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        # Decode JSON chunk fragments and stitch them together
                        chunk = json.loads(line.decode('utf-8'))
                        full_response += chunk.get("delta", "")
                        response_placeholder.markdown(full_response + "▌")
            
            # Final output update without the blinking cursor element
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend server at {BACKEND_URL}: {e}")
