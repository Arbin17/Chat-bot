import streamlit as st
import requests

st.set_page_config(page_title="Tech Chatbot", page_icon="🤖")

st.title("🤖 Tech Assistant")
st.caption("Smart chatbot powered by FAQ + DialoGPT")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender.lower()):
        st.markdown(f"**{sender}:** {msg}")

# Input from user
user_input = st.text_input("You:", key="user_input")

# If user submits input
if user_input:
    # Show user's message
    st.session_state.chat_history.append(("You", user_input))

    # Send message to FastAPI backend
    try:
        response = requests.post("http://localhost:8000/chat", json={"message": user_input})
        if response.status_code == 200:
            bot_reply = response.json().get("response", "Sorry, I couldn't process that.")
        else:
            bot_reply = "⚠️ Error: Failed to reach the chatbot server."
    except Exception as e:
        bot_reply = f"⚠️ Exception: {e}"

    # Append bot response
    st.session_state.chat_history.append(("Bot", bot_reply))

    # Clear the input field
    st.experimental_rerun = lambda: None 

# New chat button
if st.button("🧹 New Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun = lambda: None 
