import streamlit as st
import requests
import json
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BACKEND_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BACKEND_URL}/v1/chat/completions"
HISTORY_FILE = os.path.abspath("./chat_history.json")
# ==============================================================================

# Setup page layout like LM Studio
st.set_page_config(page_title="LM Studio UI Clone", layout="wide")
st.title("🤖 Local LM Studio Playground")

# Sidebar configurations
st.sidebar.header("Model Configuration")

# --- Dynamic API Server Health Check ---
backend_alive = False
engine_info = "Unknown"
ram_info = "N/A"
try:
    response = requests.get(BACKEND_URL, timeout=1.5)
    if response.status_code == 200:
        backend_alive = True
        data = response.json()
        engine_info = data.get("active_backend_engine", "Unknown")
        ram_info = data.get("backend_ram_usage_mb", "N/A")
except requests.exceptions.RequestException:
    backend_alive = False

if backend_alive:
    st.sidebar.success("● Connected to Backend")
    st.sidebar.info(f"**Engine:** {engine_info}\n\n**Backend RAM:** {ram_info}")
else:
    st.sidebar.error("○ Offline: Run backend.py")

system_prompt = st.sidebar.text_area(
    "System Prompt", 
    value="You are a helpful, concise AI assistant."
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

# Dynamic parameter entry with a maximum absolute engine cap of 8192
max_tokens = st.sidebar.number_input(
    "Max Tokens", 
    min_value=1, 
    max_value=8192, 
    value=512,
    help="Limits how long the AI response can be. 1 token ≈ 4 characters."
)

# --- Dynamic Context Guardrail Alerts ---
if max_tokens > 2048:
    st.sidebar.warning(
        f"⚠️ **High Context Warning:** {max_tokens} tokens might cause slow generation speeds on your laptop CPU. "
        "Recommended limit is ≤ 2048 tokens."
    )
else:
    st.sidebar.caption(f"💡 Target output limit: ~{int(max_tokens * 0.75)} words.")

# --- Persistence Layer Logic ---
def save_chat_to_disk(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

def load_chat_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Initialize history in session state
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_from_disk()

# --- Markdown Export Generator ---
def generate_markdown_content(messages, sys_prompt):
    md_text = f"# 🤖 LM Studio Chat Export\n\n"
    md_text += f"**System Prompt:** `{sys_prompt}`\n"
    md_text += "---\n\n"
    
    for msg in messages:
        role_label = "👤 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
        md_text += f"### {role_label}\n{msg['content']}\n\n"
    return md_text

# Sidebar Chat Management
st.sidebar.subheader("Chat Actions")

if st.session_state.messages:
    md_data = generate_markdown_content(st.session_state.messages, system_prompt)
    st.sidebar.download_button(
        label="📥 Export Chat to Markdown",
        data=md_data,
        file_name="lm_studio_chat_history.md",
        mime="text/markdown",
        use_container_width=True
    )

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.session_state.messages = []
    st.rerun()

# Display historical chat log threads in the UI layout canvas
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Lock down input control if backend isn't responding
chat_placeholder = "Type your message here..." if backend_alive else "Chat disabled: Start backend server in console first."
user_input = st.chat_input(chat_placeholder, disabled=not backend_alive)

# --- NEW: Live Word & Token Counter Calculations ---
total_words = 0
total_estimated_tokens = 0

for msg in st.session_state.messages:
    text_content = msg["content"]
    total_words += len(text_content.split())
    # Standard engineering approximation: characters divided by 4
    total_estimated_tokens += max(1, len(text_content) // 4)

# Render counter cleanly using small subtitle columns right beneath the chat input box
col1, col2, col3 = st.columns([1.5, 1.5, 7])
with col1:
    st.caption(f"📝 **Total Words:** {total_words}")
with col2:
    st.caption(f"📊 **Est. Tokens:** {total_estimated_tokens} / 8192")
with col3:
    # Alert the user if the historical context is getting near performance drop limits
    if total_estimated_tokens > 2048:
        st.caption("⚠️ *Large context history loaded. CPU latency may increase.*")
    else:
        st.caption("")

# Accept user input
if user_input:
    # Render user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_chat_to_disk(st.session_state.messages)

    # Render assistant placeholder container for real-time streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        payload = {
            "message": user_input,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": int(max_tokens)
        }

        try:
            with requests.post(CHAT_ENDPOINT, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        full_response += chunk.get("delta", "")
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat_to_disk(st.session_state.messages)
            st.rerun() 

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend server at {BACKEND_URL}: {e}")
