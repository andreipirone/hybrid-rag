import streamlit as st
from retrieval import rag_chain

def response_streamer():
    for chunk in rag_chain.stream(prompt):
        yield chunk

st.title("RAG chat (debug)")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me anything!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = rag_chain.invoke(prompt)
    response = st.chat_message("assistant").write_stream(response_streamer)
    st.session_state.messages.append({"role": "assistant", "content": response})