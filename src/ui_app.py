import streamlit as st

st.title("RAG chat (debug)")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me anything!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = "labubu"
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)