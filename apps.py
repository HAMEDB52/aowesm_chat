"""Streaming chat app built on LangChain and Streamlit."""

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = "You are a helpful assistant."
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
ROLES = {"user": HumanMessage, "assistant": AIMessage}


def api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("OPENAI_API_KEY", "")


st.set_page_config(page_title="LangChain Chat", page_icon="💬")
st.title("LangChain Chat")

with st.sidebar:
    model = st.selectbox("Model", MODELS)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    if st.button("Clear chat"):
        st.session_state.messages = []

key = api_key()
if not key:
    st.error("Set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
    st.stop()

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    history = [SystemMessage(SYSTEM_PROMPT)] + [ROLES[m["role"]](m["content"]) for m in st.session_state.messages]
    llm = ChatOpenAI(api_key=key, model=model, temperature=temperature, streaming=True)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(chunk.content for chunk in llm.stream(history))
        except Exception as exc:  # network, auth or quota errors
            st.error(f"The model call failed: {exc}")
            st.stop()
    st.session_state.messages.append({"role": "assistant", "content": reply})
